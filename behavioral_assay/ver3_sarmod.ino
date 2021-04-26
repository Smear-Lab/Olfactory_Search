int watercalL = 178; //left water port calibration in ms 
int watercalR = 232; //right water port calibration in ms
int watercalNP = 109; //initiation water port calibration in ms
//#define LED_PIN   13
//
//#define TRIGGER1   3
//#define TRIGGER2   2
//
//#define SOLENOID1 30
//#define SOLENOID2 31
//#define SOLENOID3 32
//#define SOLENOID4 33
//#define SOLENOID5 34
//#define SOLENOID6 35
//#define SOLENOID7 36
//#define SOLENOID8 37
//
//#define BEAM1     38
//#define BEAM2     39
//#define BEAM3     40
//#define BEAM4     41
//
//#define CUE1      42
//#define CUE2      43
//#define CUE3      44
//#define CUE4      45
//
//#define ADC_PIN   49
//#define DAC_PIN   48
//
//#define DIGITAL1  62
//#define DIGITAL2  63
//#define DIGITAL3  64
//#define DIGITAL4  65

#define LED_PIN   13

#define TRIGGER1   4
#define TRIGGER2   5
#define TRIGGER3   6

#define SOLENOID1 29
#define SOLENOID2 28
#define SOLENOID3 27
#define SOLENOID4 26
#define SOLENOID5 25
#define SOLENOID6 24
#define SOLENOID7 23
#define SOLENOID8 22

#define BEAM1     37
#define BEAM2     36
#define BEAM3     35
#define BEAM4     34
#define BEAM5     33
#define BEAM6     32
#define BEAM7     31
#define BEAM8     30

#define CUE1      45
#define CUE2      44
#define CUE3      43
#define CUE4      42
#define CUE5      41
#define CUE6      40
#define CUE7      39
#define CUE8      38

#define ADC_PIN   49
#define DAC1_PIN  53
#define DAC2_PIN  48
#define TEENSY_PIN 47



#define DIGITAL1  62
#define DIGITAL2  63
#define DIGITAL3  64
#define DIGITAL4  65
#define DIGITAL5  66
#define DIGITAL6  67
#define DIGITAL7  68
#define DIGITAL8  69
#define DIGITAL9  54
#define DIGITAL10 55
#define DIGITAL11 56
#define DIGITAL12 57
#define DIGITAL13 58
#define DIGITAL14 59
#define DIGITAL15 60
#define DIGITAL16 61

// include the library code:
#include <SPI.h>


char test_string[] = {"USB Fast Serial Transmit Bandwidth Test, capture this text.\r\n"};

char buffer[128], tbuffer[128];
char *argv[8];
int nosepoke = 1;
int sensorPin = A4;
int sniffamp = 69;
int pressuresensor = 68; 
int sensorValue = 0;
int arg1, arg2, arg3;
int ain;
int wateron;
uint8_t state;
uint16_t count;
uint8_t idx = 0, teensyidx = 0;

// AD7328 ADC read routine
uint16_t adcRead(uint8_t adc, uint8_t coding) {
  // adc = adc input, 0 - 7
  // coding = 0 -> two's compliment
  // coding = 1 -> binary
  uint16_t adcValue;
  digitalWrite(ADC_PIN, LOW);
  SPI.transfer(0x80 | ((adc & 0x07)<<2));
  SPI.transfer(0x10 | ((coding & 0x01)<<5));
  digitalWrite(ADC_PIN, HIGH);
  digitalWrite(ADC_PIN, LOW);
  ((uint8_t*)&adcValue)[1] = SPI.transfer(0);
  ((uint8_t*)&adcValue)[0] = SPI.transfer(0);
  digitalWrite(ADC_PIN, HIGH);
  
  // sign-extend if negative
  if ((coding == 0) && ((adcValue & 0x1000) == 0x1000)) {
    adcValue = (adcValue >> 1) | 0xf000;
  } else {
    adcValue = (adcValue >> 1) & 0x0fff;
  }
  
  // return the 12 bit value
  return adcValue;
}

// AD5666 DAC write routine
void dac1Write(uint8_t dac, uint16_t value) {
  // dac = dac output channel, 0 - 3
  // value = 16 bit output value
  digitalWrite(DAC1_PIN, LOW);
  SPI.transfer(0x03); // CMD = 0011, write & update dac channel
  SPI.transfer(((dac & 0x0f)<<4) | ((value & 0xf000)>>12));
  SPI.transfer((value & 0x0ff0)>>4);
  SPI.transfer((value & 0x000f)<<4);
  digitalWrite(DAC1_PIN, HIGH);
  digitalRead(DAC1_PIN);
}

// AD5754 DAC write routine
void dac2Write(uint8_t dac, int16_t value) {
  // dac = dac output channel, 0 - 3
  // value = 16 bit output value
  digitalWrite(DAC2_PIN, LOW);
  SPI.transfer(dac);
  SPI.transfer((value & 0xff00)>>8);
  SPI.transfer((value & 0x00ff));
  digitalWrite(DAC2_PIN, HIGH);
  digitalRead(DAC2_PIN);
}

// ******************* teensy support routines ********************

// teensy SPI transfer layer (1 byte out, 1 byte in)
uint8_t teensySPI(uint8_t value) {
  uint8_t rec;
  
  digitalWrite(TEENSY_PIN, LOW);
  rec = SPI.transfer(value);
  digitalWrite(TEENSY_PIN, HIGH);
  return rec;
}

// send 1 data byte to teensy (takes two SPI transfers)
void teensyWrite(uint8_t c) {
  teensySPI(0x81); // indicate 1 byte write transfer (0x80 + byte count)
  teensySPI(c); // send the byte
}

// send many bytes to teensy (max 127 bytes) from a buffer
// this will take n+1 SPI transactions
void teensyWriteMany(char *buff, int length) {
  uint8_t c, count, idx = 0;
  
  if (length < 128) {
    count = (length & 0x7f);
    // first send number of bytes to write
    teensySPI(length | 0x80); // multy-byte write transfer
    // then send the bytes in a loop
    while (count--) {
      c = buff[idx++];
      teensySPI(c);
    }
  }
}

// read how many bytes are available to read from teensy
// this will take two SPI transactions
uint8_t teensyAvailable() {
  teensySPI(0x00); // 0x00 -> teensy will send bytes available at next SPI transaction
  return (teensySPI(0x00)); // get the return value
}

// read a single byte from teensy
// this will take two SPI transactions
uint8_t teensyRead() {
  teensySPI(0x01); // 0x01 -> teensy will send one data byte at next SPI transaction
  return (teensySPI(0x00)); // 0x00 -> stop
}

// read many bytes from teensy into a buffer (max 255 bytes)
// if there are fewer bytes available in the teensy then the transfer count will reduced to that
// return value is the number of bytes actually read
// this will take n+1 SPI transactions if n>0
uint8_t teensyReadMany(uint8_t *buff, uint8_t length) {
  uint8_t idx = 0, avail, c;
  
  avail = teensyAvailable(); // get the number of bytes available in the teensy
  if (length < avail) // the transfer byte count is the lesser of length and avail
    avail = length;
  if (avail > 0) { // only read from teensy if we need
    teensySPI(0x01); // 0x01 -> teensy will send one data byte at next SPI transaction
    while (--avail) {
      c = teensySPI(0x01); // at least one more byte to read
      buff[idx++] = c;
    }
    c = teensySPI(0x00); // final byte
    buff[idx++] = c;
  }
  buff[idx] = 0; // terminate the buffer
  return idx; // return with byte transfer count
}

// ************************ end of teensy support routines *************************


void get_line2(char *buff, uint16_t len) {
  uint8_t c;
  uint8_t idx = 0;
  for (;;) {
    if (Serial2.available() > 0) {
      c = Serial2.read();
      if (c == '\r') break;
      if ((c >= ' ') && (idx < len - 1)) {
        buff[idx++] = c;
      }
    }
  }
  buff[idx] = 0;
}

void parse(char *line, char **argv, uint8_t maxArgs) {
  uint8_t argCount = 0;
  while (*line != '\0') {       /* if not the end of line ....... */ 
    while (*line == ',' || *line == ' ' || *line == '\t' || *line == '\n')
      *line++ = '\0';     /* replace commas and white spaces with 0    */
    *argv++ = line;          /* save the argument position     */
    argCount++;
    if (argCount == maxArgs-1)
      break;
    while (*line != '\0' && *line != ',' && *line != ' ' && 
           *line != '\t' && *line != '\n') 
      line++;             /* skip the argument until ...    */
  }
  *argv = '\0';                 /* mark the end of argument list  */
}

void setup() {
  pinMode(sniffamp, OUTPUT); //Sniff LED, DIGITAL8
  pinMode(pressuresensor, OUTPUT); //Pressure Sensor, DIGITAL7
  pinMode(LED_PIN, OUTPUT); // Green LED on the front
  pinMode(TRIGGER1, OUTPUT); // Pulse generator ch. 1 trigger
  pinMode(TRIGGER2, OUTPUT); // Pulse generator ch. 2 trigger
  pinMode(SOLENOID1, OUTPUT);
  pinMode(SOLENOID2, OUTPUT);
  pinMode(SOLENOID3, OUTPUT);
  pinMode(SOLENOID4, OUTPUT);
  pinMode(SOLENOID5, OUTPUT);
  pinMode(SOLENOID6, OUTPUT);
  pinMode(SOLENOID7, OUTPUT);
  pinMode(SOLENOID8, OUTPUT);
  pinMode(CUE1, OUTPUT);
  pinMode(CUE2, OUTPUT);
  pinMode(CUE3, OUTPUT);
  pinMode(CUE4, OUTPUT);
  pinMode(CUE5, OUTPUT);
  pinMode(CUE6, OUTPUT);
  pinMode(CUE7, OUTPUT);
  pinMode(CUE8, OUTPUT);
  pinMode(ADC_PIN, OUTPUT);
  pinMode(DAC2_PIN, OUTPUT);
  pinMode(TEENSY_PIN, OUTPUT);

  digitalWrite(LED_PIN, LOW);
  digitalWrite(TRIGGER1, LOW);
  digitalWrite(TRIGGER2, LOW);
  digitalWrite(SOLENOID1, LOW);
  digitalWrite(SOLENOID2, LOW);
  digitalWrite(SOLENOID3, LOW);
  digitalWrite(SOLENOID4, LOW);
  digitalWrite(SOLENOID5, LOW);
  digitalWrite(SOLENOID6, LOW);
  digitalWrite(SOLENOID7, LOW);
  digitalWrite(SOLENOID8, LOW);
  digitalWrite(CUE1, LOW);
  digitalWrite(CUE2, LOW);
  digitalWrite(CUE3, LOW);
  digitalWrite(CUE4, LOW);
  digitalWrite(CUE5, LOW);
  digitalWrite(CUE6, LOW);
  digitalWrite(CUE7, LOW);
  digitalWrite(CUE8, LOW);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(ADC_PIN, HIGH);
  digitalWrite(DAC2_PIN, HIGH);
  digitalWrite(TEENSY_PIN, HIGH);
  
  // initialize SPI:  
  SPI.begin(); 
  // use SPI clock mode 2
  SPI.setDataMode(SPI_MODE2);
  // set clock mode to FCLK/4
  SPI.setClockDivider(SPI_CLOCK_DIV4);

  // DAC1 (AD5666) setup
  // Setup DAC REF register
  digitalWrite(DAC1_PIN, LOW);
  SPI.transfer(0x08); // CMD = 1000
  SPI.transfer(0x00);
  SPI.transfer(0x00);
  SPI.transfer(0x01); // Standalone mode, REF on
  digitalWrite(DAC1_PIN, HIGH);
  digitalRead(DAC1_PIN); // add some time

  // Power up all four DACs
  digitalWrite(DAC1_PIN, LOW);
  SPI.transfer(0x04); // CMD = 0100
  SPI.transfer(0x00);
  SPI.transfer(0x00); // Normal operation (power-on)
  SPI.transfer(0x0f); // All four DACs
  digitalWrite(DAC1_PIN, HIGH);
  digitalRead(DAC1_PIN);

  // Set DAC outputs to 0V
  dac1Write(0, 0x0000); // 0V
  dac1Write(1, 0x0000); // 0V
  dac1Write(2, 0x0000); // 0V
  dac1Write(3, 0x0000); // 0V


  // ADC (AD7328) setup
  /* range register 1: +/- 5V range on ch0 0,1,2,3 */
  digitalWrite(ADC_PIN, LOW);
  SPI.transfer(0xaa);
  SPI.transfer(0xa0);
  digitalWrite(ADC_PIN, HIGH);
  
  /* range register 2: +/-5V range on ch 4,5,6,7 */
  digitalWrite(ADC_PIN, LOW);
  SPI.transfer(0xca);
  SPI.transfer(0xa0);
  digitalWrite(ADC_PIN, HIGH);

  /* sequence register: all sequence bits off */
  digitalWrite(ADC_PIN, LOW);
  SPI.transfer(0xe0);
  SPI.transfer(0x00);
  digitalWrite(ADC_PIN, HIGH);

  /* control register: ch 000, mode = 00, pm = 00, code = 0, ref = 1, seq = 00 */
  digitalWrite(ADC_PIN, LOW);
  SPI.transfer(0x80);
  SPI.transfer(0x10);
  digitalWrite(ADC_PIN, HIGH);

  
  // DAC2 (AD5754) setup
  /* DAC power control register (all ch + ref powered up)*/
  digitalWrite(DAC2_PIN, LOW);
  SPI.transfer(0x10);
  SPI.transfer(0x00);
  SPI.transfer(0x1f);
  digitalWrite(DAC2_PIN, HIGH);
  
  /* DAC control register (SDO turned off) */
  digitalWrite(DAC2_PIN, LOW);
  SPI.transfer(0x19);
  SPI.transfer(0x00);
  SPI.transfer(0x0d);
  digitalWrite(DAC2_PIN, HIGH);
  
  /* DAC output range register (all ch +/-5V range)*/
  digitalWrite(DAC2_PIN, LOW);
  SPI.transfer(0x0c); // all four DACs
  // 0x08 = DAC1, 0x09 = DAC2, 0x0a = DAC3, 0x0b = DAC4, 0x0c = all DACs
  SPI.transfer(0x00);
  SPI.transfer(0x03);
  // 0 = +5V range, 1 = +10V range, 2 = +10.8V range, 3 = +/-5V range
  // 4 = +/-10V range, 5 = +/- 10.8V range
  digitalWrite(DAC2_PIN, HIGH);
  // set outputs to 0V
  dac2Write(0, 0);
  dac2Write(1, 0);
  dac2Write(2, 0);
  dac2Write(3, 0);
  dac2Write(4, 0);
  dac2Write(5, 0);
  dac2Write(6, 0);
  dac2Write(7, 0);

  // PC communications
  Serial.begin(115200);
  Serial.println("* System ready *");
  
  // LCD communication
  Serial1.begin(19200); 
  Serial1.write(0x0c); // clear the display
  delay(10);
  Serial1.write(0x11); // Back-light on
  Serial1.print("* Smell light *");

  // Pulse generator communication
  Serial2.begin(115200);

  // Init done
  digitalWrite(LED_PIN, HIGH);
}



void loop() {
  digitalWrite(sniffamp, HIGH);
  digitalWrite(pressuresensor, HIGH);
  //sensorValue = (16 * adcRead(2, 0)); //12 bit read on AIN4
  //Serial.println(sensorValue);
  //if (sensorValue < 16100) {
    //dac2Write(1, 22000); //send to AOUT2 (sniff light) 
  //} else {
    //dac2Write(1,0);
  //}
  
  uint8_t c;
  if (Serial.available() > 0) { // PC communication
    c = Serial.read();
    if (c == '\r') {
      buffer[idx] = 0;
      Serial.write(c);
      Serial.println();
      parse((char*)buffer, argv, sizeof(argv));
      if (strcmp(argv[0], "solenoid") == 0) {
        // set or read a solenoid valve
        // solenoid range: 1 to 8
        if (strlen(argv[1]) > 0) {
          arg1 = atoi(argv[1]);
          if ((arg1 > 0) && (arg1 < 9)) {
            if (argv[2] == '\0') {
              switch (arg1) {
                case 1:  state = digitalRead(SOLENOID1); break;
                case 2:  state = digitalRead(SOLENOID2); break;
                case 3:  state = digitalRead(SOLENOID3); break;
                case 4:  state = digitalRead(SOLENOID4); break;
                case 5:  state = digitalRead(SOLENOID5); break;
                case 6:  state = digitalRead(SOLENOID6); break;
                case 7:  state = digitalRead(SOLENOID7); break;
                case 8:  state = digitalRead(SOLENOID8); break;
              }
              if (state == HIGH)
                Serial.println("on");
              else 
                Serial.println("off");
            } else if (strcmp(argv[2], "on") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(SOLENOID1, HIGH); break;
                case 2:  digitalWrite(SOLENOID2, HIGH); break;
                case 3:  digitalWrite(SOLENOID3, HIGH); break;
                case 4:  digitalWrite(SOLENOID4, HIGH); break;
                case 5:  digitalWrite(SOLENOID5, HIGH); break;
                case 6:  digitalWrite(SOLENOID6, HIGH); break;
                case 7:  digitalWrite(SOLENOID7, HIGH); break;
                case 8:  digitalWrite(SOLENOID8, HIGH); break;
              }
            } else if (strcmp(argv[2], "off") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(SOLENOID1, LOW); break;
                case 2:  digitalWrite(SOLENOID2, LOW); break;
                case 3:  digitalWrite(SOLENOID3, LOW); break;
                case 4:  digitalWrite(SOLENOID4, LOW); break;
                case 5:  digitalWrite(SOLENOID5, LOW); break;
                case 6:  digitalWrite(SOLENOID6, LOW); break;
                case 7:  digitalWrite(SOLENOID7, LOW); break;
                case 8:  digitalWrite(SOLENOID8, LOW); break;
              }
            } else if (strcmp(argv[2], "run") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(SOLENOID1, HIGH); delay(watercalR); digitalWrite(SOLENOID1,LOW);break;
                case 2:  digitalWrite(SOLENOID2, HIGH); delay(watercalL); digitalWrite(SOLENOID2,LOW); break;
                case 3:  digitalWrite(SOLENOID3, HIGH); delay(watercalNP); digitalWrite(SOLENOID3,LOW); break;
                case 4:  digitalWrite(SOLENOID4, HIGH); delay(watercalNP); digitalWrite(SOLENOID4,LOW);break;
                case 5:  digitalWrite(SOLENOID5, HIGH); delay(watercalNP); digitalWrite(SOLENOID5,LOW); break;
                case 6:  digitalWrite(SOLENOID6, HIGH); delay(watercalNP); digitalWrite(SOLENOID6,LOW); break;
                case 7:  digitalWrite(SOLENOID7, HIGH); delay(watercalNP); digitalWrite(SOLENOID7,LOW);break;
                case 8:  digitalWrite(SOLENOID8, HIGH); delay(watercalNP); digitalWrite(SOLENOID8,LOW); break;
              }
            } else {
              Serial.println("solenoid <N> {on,off}");
            }
          } else {
            Serial.println("solenoid <N> {on,off}, N = {1..8}");
          }
        } else {
          Serial.println("solenoid <N> {on,off}");
        }
      } else if (strcmp(argv[0], "beam") == 0) {
        // read the state of a beam break circuit
        // beam range: 1 to 4
        // return value: HIGH means no light detected, i.e. the beam is broken or no photodetector present
        if (strlen(argv[1]) > 0) {
          arg1 = atoi(argv[1]);
          if ((arg1 > 0) && (arg1 < 9)) {
            switch (arg1) {
              case 1:  state = digitalRead(BEAM1); break;
              case 2:  state = digitalRead(BEAM2); break;
              case 3:  state = digitalRead(BEAM3); break;
              case 4:  state = digitalRead(BEAM4); break;
              case 5:  state = digitalRead(BEAM5); break;
              case 6:  state = digitalRead(BEAM6); break;
              case 7:  state = digitalRead(BEAM7); break;
              case 8:  state = digitalRead(BEAM8); break;
            }
            if (state == HIGH)
              Serial.println("no light detected");
            else 
              Serial.println("light detected");
          } else {
            Serial.println("beam <N>, N = {1..8}");
          }
        } else {
          Serial.println("beam <N>");
        }
      } else if (strcmp(argv[0], "cue") == 0) {
        // set or read a cue light
        // cue light range: 1 to 4
        if (strlen(argv[1]) > 0) {
          arg1 = atoi(argv[1]);
          if ((arg1 > 0) && (arg1 < 9)) {
            if (argv[2] == '\0') {
              switch (arg1) {
                case 1:  state = digitalRead(CUE1); break;
                case 2:  state = digitalRead(CUE2); break;
                case 3:  state = digitalRead(CUE3); break;
                case 4:  state = digitalRead(CUE4); break;
                case 5:  state = digitalRead(CUE5); break;
                case 6:  state = digitalRead(CUE6); break;
                case 7:  state = digitalRead(CUE7); break;
                case 8:  state = digitalRead(CUE8); break;
              }
              if (state == HIGH)
                Serial.println("on");
              else 
                Serial.println("off");
            } else if (strcmp(argv[2], "on") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(CUE1, HIGH); break;
                case 2:  digitalWrite(CUE2, HIGH); break;
                case 3:  digitalWrite(CUE3, HIGH); break;
                case 4:  digitalWrite(CUE4, HIGH); break;
                case 5:  digitalWrite(CUE5, HIGH); break;
                case 6:  digitalWrite(CUE6, HIGH); break;
                case 7:  digitalWrite(CUE7, HIGH); break;
                case 8:  digitalWrite(CUE8, HIGH); break;
              }
            } else if (strcmp(argv[2], "off") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(CUE1, LOW); break;
                case 2:  digitalWrite(CUE2, LOW); break;
                case 3:  digitalWrite(CUE3, LOW); break;
                case 4:  digitalWrite(CUE4, LOW); break;
                case 5:  digitalWrite(CUE5, LOW); break;
                case 6:  digitalWrite(CUE6, LOW); break;
                case 7:  digitalWrite(CUE7, LOW); break;
                case 8:  digitalWrite(CUE8, LOW); break;
              }
            } else {
              Serial.println("cue <N> {on,off}");
            }
          } else {
            Serial.println("cue <N> {on,off}, N = {1..8}");
          }
        } else {
          Serial.println("cue <N> {on,off}");
        }
    
      } else if (strcmp(argv[0], "digital") == 0) {
        // set or read a digital port
        // port range: 1 to 4
        if (strlen(argv[1]) > 0) {
          arg1 = atoi(argv[1]);
          if ((arg1 > 0) && (arg1 < 9)) {
            if (argv[2] == '\0') {
              switch (arg1) {
                case 1:  state = digitalRead(DIGITAL1); break;
                case 2:  state = digitalRead(DIGITAL2); break;
                case 3:  state = digitalRead(DIGITAL3); break;
                case 4:  state = digitalRead(DIGITAL4); break;
                case 5:  state = digitalRead(DIGITAL5); break;
                case 6:  state = digitalRead(DIGITAL6); break;
                case 7:  state = digitalRead(DIGITAL7); break;
                case 8:  state = digitalRead(DIGITAL8); break;
              }
              if (state == HIGH)
                Serial.println("high");
              else 
                Serial.println("low");
            } else if (strcmp(argv[2], "high") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(DIGITAL1, HIGH); break;
                case 2:  digitalWrite(DIGITAL2, HIGH); break;
                case 3:  digitalWrite(DIGITAL3, HIGH); break;
                case 4:  digitalWrite(DIGITAL4, HIGH); break;
                case 5:  digitalWrite(DIGITAL5, HIGH); break;
                case 6:  digitalWrite(DIGITAL6, HIGH); break;
                case 7:  digitalWrite(DIGITAL7, HIGH); break;
                case 8:  digitalWrite(DIGITAL8, HIGH); break;
              }
            } else if (strcmp(argv[2], "low") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(DIGITAL1, LOW); break;
                case 2:  digitalWrite(DIGITAL2, LOW); break;
                case 3:  digitalWrite(DIGITAL3, LOW); break;
                case 4:  digitalWrite(DIGITAL4, LOW); break;
                case 5:  digitalWrite(DIGITAL5, LOW); break;
                case 6:  digitalWrite(DIGITAL6, LOW); break;
                case 7:  digitalWrite(DIGITAL7, LOW); break;
                case 8:  digitalWrite(DIGITAL8, LOW); break;
              }
            } else if (strcmp(argv[2], "input") == 0) {
              switch (arg1) {
                case 1:  pinMode(DIGITAL1, INPUT); break;
                case 2:  pinMode(DIGITAL2, INPUT); break;
                case 3:  pinMode(DIGITAL3, INPUT); break;
                case 4:  pinMode(DIGITAL4, INPUT); break;
                case 5:  pinMode(DIGITAL5, INPUT); break;
                case 6:  pinMode(DIGITAL6, INPUT); break;
                case 7:  pinMode(DIGITAL7, INPUT); break;
                case 8:  pinMode(DIGITAL8, INPUT); break;
              }
            } else if (strcmp(argv[2], "output") == 0) {
              switch (arg1) {
                case 1:  pinMode(DIGITAL1, OUTPUT); break;
                case 2:  pinMode(DIGITAL2, OUTPUT); break;
                case 3:  pinMode(DIGITAL3, OUTPUT); break;
                case 4:  pinMode(DIGITAL4, OUTPUT); break;
                case 5:  pinMode(DIGITAL5, OUTPUT); break;
                case 6:  pinMode(DIGITAL6, OUTPUT); break;
                case 7:  pinMode(DIGITAL7, OUTPUT); break;
                case 8:  pinMode(DIGITAL8, OUTPUT); break;
              }
            } else {
              Serial.println("digital <N> {high,low,input,output}");
            }
          } else {
            Serial.println("digital <N> {high,low,input,output}, N = {1..8}");
          }
        } else {
          Serial.println("digital <N> {high,low,input,output}");
        }
    
      } else if (strcmp(argv[0], "aout") == 0) {
        if ((strlen(argv[1]) > 0) && (strlen(argv[2]) > 0)) {
          arg1 = atoi(argv[1]);
          arg2 = atoi(argv[2]);
          if ((arg1 > 0) && (arg1 < 5)) {
            // write a value to an bipolar analog output port
            // port range: 1 to 4
            // output range: -5V to +5V
            // output value: 16 bit signed integer, range: -32768 to +32767
            dac2Write((uint8_t)arg1-1, arg2);
          } else if ((arg1 > 4) && (arg1 < 9)) {
            // write a value to an unipolar analog output port
            // port range: 5 to 8
            // output range: 0V to +5V
            // output value: 16 bit unsigned integer, range: 0 to 65535
            dac1Write((uint8_t)arg1-5, (uint16_t)arg2);
          } else {
            Serial.println("aout <N> <VALUE>, N = {1..8}");
          }
        } else {
          Serial.println("aout <N> <VALUE>");
        }
      } else if (strcmp(argv[0], "ain") == 0) {
        // read a value from an analog input port
        // port range: 1 to 8
        // input range: -5V to +5V
        // return value: 12 bit signed integer, range: -2048 to +2047
        if (strlen(argv[1]) > 0) {
          arg1 = atoi(argv[1]);
          if ((arg1 > 0) && (arg1 < 9)) {
            ain = adcRead((uint8_t)arg1-1, 0);
            Serial.println(ain);
          } else {
            Serial.println("ain <N>, N = {1..8}");
          }
        } else {
          Serial.println("ain <N>");
        }
      } else if (strcmp(argv[0], "teensy") == 0) {
        if (strlen(argv[1]) > 0) {
          teensyWriteMany(argv[1], strlen(argv[1]));
        }
      } else if (strcmp(argv[0], "speedtest") == 0) {
        uint16_t cnt = 0;
        uint32_t start = millis();
        while (millis() - start < 10000) {
          teensyWriteMany(test_string, strlen(test_string));
          cnt++;
        }
        Serial.print(cnt);
        Serial.print(" test strings sent in 10 seconds (");
        Serial.print((uint32_t)cnt * strlen(test_string)/10);
        Serial.println(" char/sec)");
      } else if (strcmp(argv[0], "pulsegen") == 0) {
        if (strlen(argv[1]) > 0) {
          arg1 = atoi(argv[1]);
          if ((arg1 > 0) && (arg1 < 3)) {
            if (argv[2] == '\0') {
              Serial2.println();
              delay(1);
              //Serial2.flush();
              switch (arg1) {
                case 1:  Serial2.println("r1 "); break;
                case 2:  Serial2.println("r2 "); break;
              }
              get_line2((char*)buffer, sizeof(buffer));
              parse((char*)buffer, argv, sizeof(argv));
              arg1 = atoi(argv[0]);
              arg2 = atoi(argv[1]);
              Serial.print((uint16_t)arg1);
              Serial.print(" ");
              Serial.println((uint16_t)arg2);
            } else if (strcmp(argv[2], "trigger") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(TRIGGER1, HIGH);
                         digitalWrite(TRIGGER1, LOW);
                         break;
                case 2:  digitalWrite(TRIGGER2, HIGH);
                         digitalWrite(TRIGGER2, LOW); 
                         break;
              }
            } else if (strcmp(argv[2], "set") == 0) {
              arg2 = atoi(argv[3]);
              arg3 = atoi(argv[4]);
              if (arg2 > 0) { // must be positive pulse length
              Serial2.println();
                switch (arg1) {
                  case 1:  Serial2.print("p1 "); break;
                  case 2:  Serial2.print("p2 "); break;
                }
                Serial2.print((uint16_t)arg2);
                Serial2.print(" ");
                Serial2.println((uint16_t)arg3);
              }
            }
          } else {
            Serial.println("pulsegen <N> {trigger, set <VALUE1 VALUE2>}, N = {1..2}");
          }
        } else {
          Serial.println("pulsegen <N> {trigger,set <VALUE1 VALUE2>}");
        }
      } else if ((strcmp(argv[0], "help") == 0) || (strcmp(argv[0], "?") == 0)) {
        Serial.println("BCS verification firmware");
        Serial.println("Commands:");
        Serial.println("  solenoid <N> {on,off}, N = {1..8}");
        Serial.println("  beam <N>, N = {1..8}");
        Serial.println("  cue <N> {on,off}, N = {1..8}");
        Serial.println("  digital <N> {high,low,input,output}, N = {1..8}");
        Serial.println("  aout <N> <VALUE>, N = {1..8}, -32768<VALUE<32767 for N = 1 - 4, 0 <VALUE<65535 for N = 5 - 8");
        Serial.println("  ain <N>, N = {1..8}");
        Serial.println("  pulsegen <N> {trigger, set <VALUE1 VALUE2>}, N = {1..2}, VALUE1 = time in steps of 20uS, VALUE2 = amplitude (0 - 65535)");
        Serial.println("  teensy <string>, send string to teensy");
        Serial.println("  speedtest, test teensy transmit speed");
      }
      idx = 0;
    } else if (((c == '\b') || (c == 0x7f)) && (idx > 0)) {
      idx--;
      Serial.write(c);
      Serial.print(" ");
      Serial.write(c);
    } 
    else if ((c >= ' ') && (idx < sizeof(buffer) - 1)) {
      buffer[idx++] = c;
      Serial.write(c);
    }
  }
  if (teensyAvailable() > 0) {
    c = teensyRead();
    if (c == '\r') {
      tbuffer[teensyidx] = 0;
      teensyWrite(c);
      teensyWrite('\n');
      parse((char*)tbuffer, argv, sizeof(argv));
      // add one sample command just to test the teensy command line functionallity
      if (strcmp(argv[0], "solenoid") == 0) {
        // set or read a solenoid valve
        // solenoid range: 1 to 8
        if (strlen(argv[1]) > 0) {
          arg1 = atoi(argv[1]);
          if ((arg1 > 0) && (arg1 < 9)) {
            if (argv[2] == '\0') {
              switch (arg1) {
                case 1:  state = digitalRead(SOLENOID1); break;
                case 2:  state = digitalRead(SOLENOID2); break;
                case 3:  state = digitalRead(SOLENOID3); break;
                case 4:  state = digitalRead(SOLENOID4); break;
                case 5:  state = digitalRead(SOLENOID5); break;
                case 6:  state = digitalRead(SOLENOID6); break;
                case 7:  state = digitalRead(SOLENOID7); break;
                case 8:  state = digitalRead(SOLENOID8); break;
              }
              if (state == HIGH) {
                sprintf(tbuffer, "on \r\n");
                teensyWriteMany((char*)tbuffer, strlen(tbuffer));
              } else {
                sprintf(tbuffer, "off \r\n");
                teensyWriteMany((char*)tbuffer, strlen(tbuffer));
              }
            } else if (strcmp(argv[2], "on") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(SOLENOID1, HIGH); break;
                case 2:  digitalWrite(SOLENOID2, HIGH); break;
                case 3:  digitalWrite(SOLENOID3, HIGH); break;
                case 4:  digitalWrite(SOLENOID4, HIGH); break;
                case 5:  digitalWrite(SOLENOID5, HIGH); break;
                case 6:  digitalWrite(SOLENOID6, HIGH); break;
                case 7:  digitalWrite(SOLENOID7, HIGH); break;
                case 8:  digitalWrite(SOLENOID8, HIGH); break;
              }
            } else if (strcmp(argv[2], "off") == 0) {
              switch (arg1) {
                case 1:  digitalWrite(SOLENOID1, LOW); break;
                case 2:  digitalWrite(SOLENOID2, LOW); break;
                case 3:  digitalWrite(SOLENOID3, LOW); break;
                case 4:  digitalWrite(SOLENOID4, LOW); break;
                case 5:  digitalWrite(SOLENOID5, LOW); break;
                case 6:  digitalWrite(SOLENOID6, LOW); break;
                case 7:  digitalWrite(SOLENOID7, LOW); break;
                case 8:  digitalWrite(SOLENOID8, LOW); break;
              }
            } else {
              sprintf(tbuffer, "solenoid <N> {on,off}\r\n");
              teensyWriteMany((char*)tbuffer, strlen(tbuffer));
            }
          } else {
            sprintf(tbuffer, "solenoid <N> {on,off}, N = {1..8}\r\n");
            teensyWriteMany((char*)tbuffer, strlen(tbuffer));
          }
        } else {
          sprintf(tbuffer, "solenoid <N> {on,off}\r\n");
          teensyWriteMany((char*)tbuffer, strlen(tbuffer));
        }
      }
      teensyidx = 0;
    } else if (((c == '\b') || (c == 0x7f)) && (teensyidx > 0)) {
      teensyidx--;
      teensyWrite(c);
      teensyWrite(' ');
      teensyWrite(c);
    } 
    else if ((c >= ' ') && (teensyidx < sizeof(tbuffer) - 1)) {
      tbuffer[teensyidx++] = c;
      teensyWrite(c);
    }
  }

if (nosepoke == 0){ 
  if (digitalRead(BEAM1) == 1){
   Serial.println(3); nosepoke = 1;
   dac2Write(3, 21000); //send to AOUT2 (initiation port)
  }
  if (digitalRead(BEAM2) == 1){
   Serial.println(2); nosepoke = 1;
   dac2Write(3, 14000); //send to AOUT2 (left port)
  }
  if (digitalRead(BEAM3) == 1){
   Serial.println(1); nosepoke = 1;
   dac2Write(3, 7000); //send to AOUT2 (right port) 
  }
  }
if (nosepoke == 1){
  if (digitalRead(BEAM1) == 0){
    if (digitalRead(BEAM2) == 0){
      if (digitalRead(BEAM3) == 0){
        Serial.println(0); nosepoke = 0;
        dac2Write(3, 0); //send to AOUT2 (no port)
      }}}}
delay(10);
}
