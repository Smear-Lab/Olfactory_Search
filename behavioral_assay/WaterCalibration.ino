int type_calibration = 3;
//type_calibration = 0 means flushing out system
//type_calibration  = 1 means calibrating right water port
//type_calibration  = 2 means calibrating left water port
//type_calibration  = 3 means calibrating initiation water port
//type-calibration  = 4 means calibrate all ports at once

 int training = 1; 
 //training = 0 means you are not doing trainer 1 today
 //training = 1 means you are doing trainer 1 today

int watercalL = 178; //left water port calibration in ms 
int watercalR = 232; //right water port calibration in ms
int watercalNP = 109; //initiation water port calibration in ms
 //This is to change the time open for each iteration of water valves opening
 //To change water calibration run time, refer to this variable below: (type_calibration == #)
 

// command definitions
#define SOLENOID1 29
#define SOLENOID2 28
#define SOLENOID3 27
#define SOLENOID4 26
#define SOLENOID5 25
#define SOLENOID6 24
#define SOLENOID7 23
#define SOLENOID8 22
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

int i = 1; int watercalF = 500; //in ms
int delaybetweenwater = 500;
boolean state = true;

void setup()
{
  pinMode(SOLENOID4, OUTPUT);
  pinMode(SOLENOID1, OUTPUT);
  pinMode(SOLENOID2, OUTPUT);
  digitalWrite(SOLENOID4, LOW);
  digitalWrite(SOLENOID1, LOW);
  digitalWrite(SOLENOID2, LOW);
 
}
void loop() {
  if (state == true){
    if (type_calibration == 0){
      watercalF = 500;
      for (int i = 0; i < 50; i++) {
        delay(delaybetweenwater); //time between opening of valve (msec)
        if (training == 1) {
          digitalWrite(SOLENOID4, HIGH); //Water BNP On
        }
        digitalWrite(SOLENOID1, HIGH); //Water Right On
        digitalWrite(SOLENOID2, HIGH); //Water Left On
        delay(watercalF); // time in msec that valve is open
        if (training == 1) {
          digitalWrite(SOLENOID4, LOW); //Water BNP Off
        }
        digitalWrite(SOLENOID1, LOW); //Water Right Off
        digitalWrite(SOLENOID2, LOW); //Water Left On
      }}  
    if (type_calibration == 1){
      for (int i = 0; i < 100; i++) {
        delay(delaybetweenwater); //time between opening of valve (msec)
        digitalWrite(SOLENOID1, HIGH); //Water Right On
        delay(watercalR); // time in msec that valve is open
        digitalWrite(SOLENOID1, LOW); //Water Right Off
      }}  
    if (type_calibration == 2){
      for (int i = 0; i < 100; i++) {
        delay(delaybetweenwater); //time between opening of valve (msec)
        digitalWrite(SOLENOID2, HIGH); //Water Left On
        delay(watercalL); // time in msec that valve is open
        digitalWrite(SOLENOID2, LOW); //Water Left Off
      }}  
    if (type_calibration == 3){
      for (int i = 0; i < 100; i++) {
        delay(delaybetweenwater); //time between opening of valve (msec)
        digitalWrite(SOLENOID4, HIGH); //Water BNP On
        delay(watercalNP); // time in msec that valve is open
        digitalWrite(SOLENOID4, LOW); //Water BNP Off
      }}
    if (type_calibration == 4){
      for (int i = 0; i < 100; i++) {
        delay(200); //time between opening the valves (msec)
        digitalWrite(SOLENOID1, HIGH); //Water Right On
        delay(watercalR); // time in msec that valve is open
        digitalWrite(SOLENOID1, LOW); //Water Right Off
        digitalWrite(SOLENOID2, HIGH); //Water Left On
        delay(watercalL); // time in msec that valve is open
        digitalWrite(SOLENOID2, LOW); //Water Left Off
        if (training == 1){
          digitalWrite(SOLENOID4, HIGH); //Water BNP On
          delay(watercalNP); // time in msec that valve is open
          digitalWrite(SOLENOID4, LOW); //Water BNP Off
        }
      }}  
  }
    state = false;
  }
// end of loop
      
