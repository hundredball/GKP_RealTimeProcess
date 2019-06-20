"""
Core OpenBCI object for handling connections and samples from the board.

EXAMPLE USE:

def handle_sample(sample):
  print(sample.channel_data)

board = OpenBCIBoard()
board.print_register_settings()
board.start_streaming(handle_sample)

NOTE: If daisy modules is enabled, the callback will occur every two samples, hence "packet_id" will only contain even numbers. As a side effect, the sampling rate will be divided by 2.

FIXME: at the moment we can just force daisy mode, do not check that the module is detected.
TODO: enable impedance

"""
import serial
import struct
import numpy as np
import time
import timeit
import atexit
import logging
import threading
import sys
import pdb
import glob
import socket
import h5py
import os
import os.path
import torch #190329
from pathlib import Path
import Model

SAMPLE_RATE = 250.0  # Hz
START_BYTE = 0xA0  # start of data packet
END_BYTE = 0xC0  # end of data packet
ADS1299_Vref = 4.5  #reference voltage for ADC in ADS1299.  set by its hardware
ADS1299_gain = 24.0  #assumed gain setting for ADS1299.  set by its Arduino code
scale_fac_uVolts_per_count = ADS1299_Vref/float((pow(2,23)-1))/ADS1299_gain*1000000.
scale_fac_accel_G_per_count = 0.002 /(pow(2,4)) #assume set to +/4G, so 2 mG 
'''
#Commands for in SDK http://docs.openbci.com/software/01-Open BCI_SDK:

command_stop = "s";
command_startText = "x";
command_startBinary = "b";
command_startBinary_wAux = "n";
command_startBinary_4chan = "v";
command_activateFilters = "F";
command_deactivateFilters = "g";
command_deactivate_channel = {"1", "2", "3", "4", "5", "6", "7", "8"};
command_activate_channel = {"q", "w", "e", "r", "t", "y", "u", "i"};
command_activate_leadoffP_channel = {"!", "@", "#", "$", "%", "^", "&", "*"};  //shift + 1-8
command_deactivate_leadoffP_channel = {"Q", "W", "E", "R", "T", "Y", "U", "I"};   //letters (plus shift) right below 1-8
command_activate_leadoffN_channel = {"A", "S", "D", "F", "G", "H", "J", "K"}; //letters (plus shift) below the letters below 1-8
command_deactivate_leadoffN_channel = {"Z", "X", "C", "V", "B", "N", "M", "<"};   //letters (plus shift) below the letters below the letters below 1-8
command_biasAuto = "`";
command_biasFixed = "~";
'''


class OpenBCICyton(object):
  """

  Handle a connection to an OpenBCI board.

  Args:
    port: The port to connect to.
    baud: The baud of the serial connection.
    daisy: Enable or disable daisy module and 16 chans readings
    aux, impedance: unused, for compatibility with ganglion API
  """

  def __init__(self, port=None, baud=115200, filter_data=True,
    scaled_output=True, daisy=False, aux=False, impedance=False, log=True, num=0, timeout=None):
    self.log = log # print_incoming_text needs log
    self.streaming = False
    self.baudrate = baud
    self.timeout = timeout
    if not port:
        port = self.find_port()
    self.port = port
    # might be handy to know API
    self.board_type = "cyton"
    print("Connecting to V3 at port %s" %(port))
    self.ser = serial.Serial(port= port, baudrate = baud, timeout=timeout)
    #baudrate 波特率 timeout 讀超時設置

    print("Serial established...")

    time.sleep(2)
    #Initialize 32-bit board, doesn't affect 8bit board
    self.ser.write(b'v');
    #向端口寫數據

    #wait for device to be ready
    time.sleep(1)
    self.print_incoming_text()
    self.streaming = False
    self.filtering_data = filter_data
    self.scaling_output = scaled_output
    self.eeg_channels_per_sample = 8 # number of EEG channels per sample *from the board*
    self.aux_channels_per_sample = 3 # number of AUX channels per sample *from the board*
    self.imp_channels_per_sample = 0 # impedance check not supported at the moment
    self.read_state = 0
    self.daisy = daisy
    self.last_odd_sample = OpenBCISample(-1, [], []) # used for daisy
    self.log_packet_count = 0
    self.attempt_reconnect = False
    self.last_reconnect = 0
    self.reconnect_freq = 5
    self.packets_dropped = 0
    self.records = np.expand_dims(np.zeros(16), axis=1)
    self.event = 0
    self.num = num
    self.filename = 'recording%s.hdf5'%self.num
    self.dset = None
    self.trial_num = 0
    self.server = None
    self.save_first()
    
    #19/03/29
    self.rt_record = np.expand_dims(np.zeros(16), axis=0)
    
    #19/03/29 loading model first, prevent repetitively load model
    local_ip = self._get_local_ip_address()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Opened socket on %s" % (local_ip))
    server.bind((local_ip, 100))
    server.listen(5)
    server.setblocking(1)
    self.s_socket, addr = server.accept() 
    
    
    m = 'EEGNet_ReLU_0420.pt'
    param = 'param_0420.txt'
    self.model = self.init_model(m, param)
    
    #Disconnects from board when terminated
    atexit.register(self.disconnect)
    #註冊程序退出時的回掉函數
  def getBoardType(self):
    """ Returns the version of the board """
    return self.board_type
  
  def setImpedance(self, flag):
    """ Enable/disable impedance measure. Not implemented at the moment on Cyton. """
    return
  
  def ser_write(self, b):
    """Access serial port object for write""" 
    self.ser.write(b)

  def ser_read(self):
    """Access serial port object for read""" 
    return self.ser.read()

  def ser_inWaiting(self):
    """Access serial port object for inWaiting"""
     #返回接收緩存忠的字節數
    return self.ser.inWaiting();

    
  def getSampleRate(self):
    if self.daisy:
      return SAMPLE_RATE/2
    else:
      return SAMPLE_RATE
  
  def getNbEEGChannels(self):
    if self.daisy:
      return self.eeg_channels_per_sample*2 #eeg chaneel # = 16
    else:
      return self.eeg_channels_per_sample #eeg channel # = 8
  
  def getNbAUXChannels(self):
    return  self.aux_channels_per_sample

  def getNbImpChannels(self):
    return  self.imp_channels_per_sample

  def start_streaming(self, callback, lapse=-1):
    #self.trigger()
    
    
    '''
    Start handling streaming data from the board. Call a provided callback
    for every single sample that is processed (every two samples?? with daisy module).

    Args:
      callback: A callback function?? -- or a list of functions -- that will receive a single argument?? of the
          OpenBCISample object captured.
      '''
    if not self.streaming:
      self.ser.write(b'b')
      self.streaming = True
    #開始時間:measure wall clock time ,not the CPU time
    start_time = timeit.default_timer()

    # Enclose callback funtion in a list if it comes alone
    #class generation: instance->metaclass的_call_method
    #def_call_(cls,*args,**kwargs)
    if not isinstance(callback, list):
       callback = [callback] #callback is not a list
    

    #Initialize check connection
    self.check_connection()

    time_point = 6  # time_point = 
    isRestMode = True
    printNow = True
    while self.streaming:
      # read current sample
      sample = self._read_serial_binary()
      #print(sample)
      if printNow:
        if time_point == 6:
          time_point = 0
          isRestMode = True
          print('---Rest---')
          self.s_socket.sendall(bytes([99]))
        else:
          print('***Move***')
          self.s_socket.sendall(bytes([100]))
        printNow = False
        
      # if a daisy module is attached, wait to concatenate two samples (main board + daisy) before passing it to callback
      if self.daisy:
        # odd sample: daisy sample, save for later
        if ~sample.id % 2: #最後一個偶數
          #print(sample.id,~sample.id)
          self.last_odd_sample = sample
        # even sample: concatenate and send if /last sample was the fist part, otherwise drop the packet
        elif sample.id - 1 == self.last_odd_sample.id:#sample.id is odd
          # the aux data will be the average between the two samples, as the channel samples themselves have been averaged by the board
          avg_aux_data = list((np.array(sample.aux_data) + np.array(self.last_odd_sample.aux_data))/2)#acc data
          whole_sample = OpenBCISample(sample.id, sample.channel_data + self.last_odd_sample.channel_data, avg_aux_data)
          #print(sample.id, sample.channel_data,self.last_odd_sample.channel_data)
          self.records = np.hstack((self.records, np.expand_dims(np.array(sample.channel_data + self.last_odd_sample.channel_data),axis=1)))
          #print(self.records.shape)
          for call in callback:
              call(whole_sample)
              
          #19/03/29 check records size, if size = 250 then call processing thread
          #print(self.rt_record, np.array(sample.channel_data + self.last_odd_sample.channel_data))
          self.rt_record = np.append(self.rt_record, np.expand_dims(np.array(sample.channel_data + self.last_odd_sample.channel_data),axis=0), axis=0)
          #print(self.rt_record.shape)
          if(self.rt_record.shape[0] == 251):
            
            if isRestMode:
              modelThread = threading.Thread(target = self.model.runThread, args=(isRestMode, self.rt_record.copy()))
              isRestMode = False
            elif ~isRestMode:
              modelThread = threading.Thread(target=self.model.runThread, args=(isRestMode, self.rt_record.copy()))
            modelThread.daemon = True # will stop on exit
            try:
                modelThread.start()
            except:
                raise
            
            self.rt_record = np.expand_dims(np.zeros(16), axis=0)
            time_point += 1
            modelThread.join()
            printNow = True
            
      else:
        for call in callback:
          call(sample)
      #lapse: record time limit
      if lapse > 0 and timeit.default_timer() - start_time > lapse:
        self.stop();
      if self.log:
        self.log_packet_count = self.log_packet_count + 1;
  def trigger(self):
      print('In the trigger')
      local_ip = self._get_local_ip_address()
      #print('local_ip',local_ip)
      self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      print("Opened first socket on %s" % (local_ip))
      self.server.bind((local_ip,80))
      self.server.listen(5)
      self.server.setblocking(1)
      while True:
           #print('here')
           client, address = self.server.accept()
           data = client.recv(1)
           if not data:
              pass
           else:
              print("First client send: " + str(data))
              client.close()
              break
  # ------- user.py main thread task ---------
  def signal(self):
    print("In the signal")
    local_ip = self._get_local_ip_address()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Opened second socket on %s" % (local_ip))
    server.bind((local_ip, 100))
    server.listen(5)
    server.setblocking(1)
    client, addr = server.accept()
    while True:
        data = client.recv(1)
        if data == b'\x00':
          print('Trial num : %d'%(self.trial_num))
          print("Second client send: " + str(data))
          self.save_record()
          self.trial_num += 1
        if data == b'\x01':
          print("Second client send: " + str(data))
          self.stop()
          client.close()
          print('finish exp')
          break
  
  def init_model(self, p, paramPath): #19/03/29
    print('Initialize model object')
    # create server for sending signal to unity, unity acts as client
#    unity_host, unity_port = '87.87.87.87', 8787
#    unity_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    unity_sock.connect((unity_host, unity_port))
#    print("In the init_model")
#    local_ip = self._get_local_ip_address()
#    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    print("Opened socket on %s" % (local_ip))
#    server.bind((local_ip, 100))
#    server.listen(5)
#    server.setblocking(1)
    #s_socket, addr = server.accept() 
#    s_socket = None
    return Model.Model(path=p, paramPath = paramPath, socket=self.s_socket)
      
      
  # ----------------------------------------
  def _get_local_ip_address(self):
      s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      s.connect(("8.8.8.8", 80))
      local_ip_address = s.getsockname()[0]
      s.close()
      return local_ip_address

  def save_first(self):
    if os.path.isfile(Path(self.filename)):
       os.remove(self.filename)
    with h5py.File(self.filename,'w') as file:
      self.dset = file.create_dataset('dataset', (16,1), maxshape=(16,None))
      self.dset[:] = self.records
      events_file = file.create_dataset('events',(1,1),maxshape=(1,None))
      events_file[:] = self.event
      print(self.dset.shape)

  def save_record(self):
    print('save recording at trial %d...'%self.trial_num)
    print('record size', self.records.shape)
    with h5py.File(self.filename, 'a') as file:
      self.dset = file['dataset']
      events_file = file['events']
      self.dset.resize((16,self.dset.shape[1]+self.records.shape[1]-1))
      events_file.resize((1, events_file.shape[1] + 1))
      print('dset size',self.dset.shape)
      #print('interval size',self.records[:,1:].shape)
      print('events_file size', events_file.shape)
      self.dset[:,-(self.records.shape[1]-1):] = self.records[:,1:]
      self.event = self.dset.shape[1]
      events_file[:,-1] = self.event
    #print('event time',self.event)
    self.records = np.expand_dims(np.zeros(16),axis=1)

  
  """
    PARSER:
    Parses incoming data packet into OpenBCISample.
    Incoming Packet Structure:
    Start Byte(1)|Sample ID(1)|Channel Data(24)|Aux Data(6)|End Byte(1)
    0xA0|0-255|8, 3-byte signed ints|3 2-byte signed ints|0xC0

  """
  def _read_serial_binary(self, max_bytes_to_skip=3000):
    def read(n):#read up to n bytes(timeout)
      bb = self.ser.read(n)
      if not bb: #bb is none
        self.warn('Device appears to be stalled. Quitting...')
        sys.exit()
        raise Exception('Device Stalled')
        sys.exit()
        return '\xFF'
      else:
        return bb #read up n bytes

    for rep in range(max_bytes_to_skip):

      #---------Start Byte & ID---------
      if self.read_state == 0:
        
        b = read(1) #read 1 byte
        
        if struct.unpack('B', b)[0] == START_BYTE: #unpack from the buffer
          if(rep != 0):
            self.warn('Skipped %d bytes before start found' %(rep))
            rep = 0;
          packet_id = struct.unpack('B', read(1))[0] #packet id goes from 0-255
          log_bytes_in = str(packet_id);

          self.read_state = 1

      #---------Channel Data---------
      elif self.read_state == 1:
        channel_data = []
        for c in range(self.eeg_channels_per_sample):

          # 3 byte ints
          literal_read = read(3)

          unpacked = struct.unpack('3B', literal_read)#3  binary number?
          log_bytes_in = log_bytes_in + '|' + str(literal_read)

          # 3byte int in 2s compliment
          if (unpacked[0] > 127):
            pre_fix = bytes(bytearray.fromhex('FF')) #255
          else:
            pre_fix = bytes(bytearray.fromhex('00')) # 0


          literal_read = pre_fix + literal_read

          # unpack little endian(>) rightmost? signed integer(i) (makes unpacking platform independent)
          myInt = struct.unpack('>i', literal_read)[0]

          if self.scaling_output:#調整signal?
            channel_data.append(myInt*scale_fac_uVolts_per_count)
          else:
            channel_data.append(myInt)

        self.read_state = 2

      #---------Accelerometer Data---------
      elif self.read_state == 2:
        aux_data = []
        for a in range(self.aux_channels_per_sample):

          #short = h,>:big-endian,<:little-endian,h:short integer(standard size:2)
          acc = struct.unpack('>h', read(2))[0]
          log_bytes_in = log_bytes_in + '|' + str(acc);

          if self.scaling_output:
            aux_data.append(acc*scale_fac_accel_G_per_count)
          else:
              aux_data.append(acc)

        self.read_state = 3;
      #---------End Byte---------
      elif self.read_state == 3:
        val = struct.unpack('B', read(1))[0]#B:unsigned char(standard size:1)
        log_bytes_in = log_bytes_in + '|' + str(val);
        self.read_state = 0 #read next packet
        if (val == END_BYTE):
          sample = OpenBCISample(packet_id, channel_data, aux_data)
          self.packets_dropped = 0
          return sample
        else:
          self.warn("ID:<%d> <Unexpected END_BYTE found <%s> instead of <%s>"      
            %(packet_id, val, END_BYTE))
          logging.debug(log_bytes_in);
          self.packets_dropped = self.packets_dropped + 1
  
  """

  Clean Up (atexit)

  """
  def stop(self):
    print("Stopping streaming...\nWait for buffer to flush...")
    self.streaming = False
    self.ser.write(b's')
    self.save_record()
    if self.log:
      logging.warning('sent <s>: stopped streaming')

  def disconnect(self):
    if(self.streaming == True):
      self.stop()
    if (self.ser.isOpen()):
      print("Closing Serial...")
      self.ser.close()
      logging.warning('serial closed')
       

  """

      SETTINGS AND HELPERS

  """
  def warn(self, text):
    if self.log:
      #log how many packets where sent succesfully in between warnings
      if self.log_packet_count:
        #logging.info('Data packets received:'+str(self.))
        self.log_packet_count = 0
      logging.warning(text)
    print("Warning: %s" % text)


  def print_incoming_text(self):
    """

    When starting the connection, print all the debug data until
    we get to a line with the end sequence '$$$'.

    """
    line = ''
    #Wait for device to send data
    time.sleep(1)
    
    if self.ser.inWaiting():
        #返回接收緩存中的字節數
      line = ''
      c = ''
     #Look for end sequence $$$
      while '$$$' not in line:
        c = self.ser.read().decode('utf-8', errors='replace') # we're supposed to get UTF8 text, but the board might behave otherwise
        #從串口讀size=1個字節
        line += c
      print('line',line);
    else:
      self.warn("No Message")

  def openbci_id(self, serial):
    """

    When automatically detecting port, parse the serial return for the "OpenBCI" ID.

    """
    line = ''
    #Wait for device to send data
    time.sleep(2)
    
    if serial.inWaiting():
      line = ''
      c = ''
     #Look for end sequence $$$
      while '$$$' not in line:
        c = serial.read().decode('utf-8', errors='replace') # we're supposed to get UTF8 text, but the board might behave otherwise
        line += c
      if "OpenBCI" in line:
        return True
    return False

  def print_register_settings(self):
    self.ser.write(b'?')
    time.sleep(0.5)
    self.print_incoming_text();
  #DEBBUGING: Prints individual incoming bytes
  def print_bytes_in(self):
    if not self.streaming:
      self.ser.write(b'b')
      self.streaming = True
    while self.streaming:
      print(struct.unpack('B',self.ser.read())[0]);

      '''Incoming Packet Structure:
    Start Byte(1)|Sample ID(1)|Channel Data(24)|Aux Data(6)|End Byte(1)
    0xA0|0-255|8, 3-byte signed ints|3 2-byte signed ints|0xC0'''

  def print_packets_in(self):
    while self.streaming:
      b = struct.unpack('B', self.ser.read())[0];
      
      if b == START_BYTE:
        self.attempt_reconnect = False
        if skipped_str:
          logging.debug('SKIPPED\n' + skipped_str + '\nSKIPPED')
          skipped_str = ''

        packet_str = "%03d"%(b) + '|';
        b = struct.unpack('B', self.ser.read())[0];
        packet_str = packet_str + "%03d"%(b) + '|';
        
        #data channels
        for i in range(24-1):
          b = struct.unpack('B', self.ser.read())[0];
          packet_str = packet_str + '.' + "%03d"%(b);

        b = struct.unpack('B', self.ser.read())[0];
        packet_str = packet_str + '.' + "%03d"%(b) + '|';

        #aux channels
        for i in range(6-1):
          b = struct.unpack('B', self.ser.read())[0];
          packet_str = packet_str + '.' + "%03d"%(b);
        
        b = struct.unpack('B', self.ser.read())[0];
        packet_str = packet_str + '.' + "%03d"%(b) + '|';

        #end byte
        b = struct.unpack('B', self.ser.read())[0];
        
        #Valid Packet
        if b == END_BYTE:
          packet_str = packet_str + '.' + "%03d"%(b) + '|VAL';
          print(packet_str)
          #logging.debug(packet_str)
        
        #Invalid Packet
        else:
          packet_str = packet_str + '.' + "%03d"%(b) + '|INV';
          #Reset
          self.attempt_reconnect = True
          
      
      else:
        print(b)
        if b == END_BYTE:
          skipped_str = skipped_str + '|END|'
        else:
          skipped_str = skipped_str + "%03d"%(b) + '.'

      if self.attempt_reconnect and (timeit.default_timer()-self.last_reconnect) > self.reconnect_freq:
        self.last_reconnect = timeit.default_timer()
        self.warn('Reconnecting')
        self.reconnect()
 


  def check_connection(self, interval = 2, max_packets_to_skip=10):
    # stop checking when we're no longer streaming
    if not self.streaming:
      return
    #check number of dropped packages and establish connection problem if too large
    if self.packets_dropped > max_packets_to_skip:
      #if error, attempt to reconect
      self.reconnect()
    # check again again in 2 seconds
    threading.Timer(interval, self.check_connection).start()

  def reconnect(self):
    self.packets_dropped = 0
    self.warn('Reconnecting')
    self.stop()
    time.sleep(0.5)
    self.ser.write(b'v')
    time.sleep(0.5)
    self.ser.write(b'b')
    time.sleep(0.5)
    self.streaming = True
    #self.attempt_reconnect = False


  #Adds a filter at 60hz to cancel out ambient electrical noise
  def enable_filters(self):
    self.ser.write(b'f')
    self.filtering_data = True;

  def disable_filters(self):
    self.ser.write(b'g')
    self.filtering_data = False;

  def test_signal(self, signal):
    """ Enable / disable test signal """
    if signal == 0:
      self.ser.write(b'0')
      self.warn("Connecting all pins to ground")
    elif signal == 1:
      self.ser.write(b'p')
      self.warn("Connecting all pins to Vcc")
    elif signal == 2:
      self.ser.write(b'-')
      self.warn("Connecting pins to low frequency 1x amp signal")
    elif signal == 3:
      self.ser.write(b'=')
      self.warn("Connecting pins to high frequency 1x amp signal")
    elif signal == 4:
      self.ser.write(b'[')
      self.warn("Connecting pins to low frequency 2x amp signal")
    elif signal == 5:
      self.ser.write(b']')
      self.warn("Connecting pins to high frequency 2x amp signal")
    else:
      self.warn("%s is not a known test signal. Valid signals go from 0-5" %(signal))

  def set_channel(self, channel, toggle_position):
    """ Enable / disable channels """
    #Commands to set toggle to on position
    if toggle_position == 1:
      if channel is 1:
        self.ser.write(b'!')
      if channel is 2:
        self.ser.write(b'@')
      if channel is 3:
        self.ser.write(b'#')
      if channel is 4:
        self.ser.write(b'$')
      if channel is 5:
        self.ser.write(b'%')
      if channel is 6:
        self.ser.write(b'^')
      if channel is 7:
        self.ser.write(b'&')
      if channel is 8:
        self.ser.write(b'*')
      if channel is 9 and self.daisy:
        self.ser.write(b'Q')
      if channel is 10 and self.daisy:
        self.ser.write(b'W')
      if channel is 11 and self.daisy:
        self.ser.write(b'E')
      if channel is 12 and self.daisy:
        self.ser.write(b'R')
      if channel is 13 and self.daisy:
        self.ser.write(b'T')
      if channel is 14 and self.daisy:
        self.ser.write(b'Y')
      if channel is 15 and self.daisy:
        self.ser.write(b'U')
      if channel is 16 and self.daisy:
        self.ser.write(b'I')
    #Commands to set toggle to off position
    elif toggle_position == 0:
      if channel is 1:
        self.ser.write(b'1')
      if channel is 2:
        self.ser.write(b'2')
      if channel is 3:
        self.ser.write(b'3')
      if channel is 4:
        self.ser.write(b'4')
      if channel is 5:
        self.ser.write(b'5')
      if channel is 6:
        self.ser.write(b'6')
      if channel is 7:
        self.ser.write(b'7')
      if channel is 8:
        self.ser.write(b'8')
      if channel is 9 and self.daisy:
        self.ser.write(b'q')
      if channel is 10 and self.daisy:
        self.ser.write(b'w')
      if channel is 11 and self.daisy:
        self.ser.write(b'e')
      if channel is 12 and self.daisy:
        self.ser.write(b'r')
      if channel is 13 and self.daisy:
        self.ser.write(b't')
      if channel is 14 and self.daisy:
        self.ser.write(b'y')
      if channel is 15 and self.daisy:
        self.ser.write(b'u')
      if channel is 16 and self.daisy:
        self.ser.write(b'i')

  def find_port(self):
    # Finds the serial port names
    if sys.platform.startswith('win'):
      ports = ['COM%s' % (i+1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
      #ports = glob.glob('/dev/ttyUSB*')
      ports = ['/dev/ttyUSB0']
    elif sys.platform.startswith('darwin'):
      ports = glob.glob('/dev/tty.usbserial*')
    else:
      raise EnvironmentError('Error finding ports on your operating system')
    openbci_port = ''
    for port in ports:
      try:
        s = serial.Serial(port= port, baudrate = self.baudrate, timeout=self.timeout)
        print(port)
        s.write(b'v')
        openbci_serial = self.openbci_id(s)
        s.close()
        if openbci_serial:
          openbci_port = port;
      except OSError as e:
        print('OSError', type(e), str(e))
        pass
      except serial.SerialException:
        print('serial_except')
        pass
    if openbci_port == '':
      raise OSError('Cannot find OpenBCI port')
    else:
      return openbci_port

class OpenBCISample(object):
  """Object encapulsating a single sample from the OpenBCI board. NB: dummy imp for plugin compatiblity"""
  def __init__(self, packet_id, channel_data, aux_data):
    self.id = packet_id
    self.channel_data = channel_data
    self.aux_data = aux_data
    self.imp_data = []


