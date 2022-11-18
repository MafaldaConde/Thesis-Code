#include <Servo.h>
Servo servoX;
Servo servoY;


bool setup_pos = true;
int p_y ;
int p_x ;
int pos = 0;
/*const int led=4;// Declaration Pin 3 for LED*/
/*const int button=3;// Declaration Pin 3 for button
int stateb=0;// Initialisation the state of the button */
char input; //serial input is stored in this variable


void setup_servos(int p_x, int p_y){
  if (setup_pos == true){
    Serial.println("Setting up servos ...");
    servoX.write(p_x);
    servoY.write(p_y);
    delay(50);
    Serial.println("Setup completed.");
    setup_pos = false;
  }
}
  
void setup() {
  Serial.begin(9600);
  Serial.setTimeout(1);
  // Input/output Configuration 
  /*pinMode(led,OUTPUT);
  pinMode(button,INPUT_PULLUP);*/
  servoX.attach(3);
  servoY.attach(6);
  setup_servos(100,63.5);
}

void loop() {

 if(Serial.available()){ //checks if any data is in the serial buffer
    int value2;
    input = Serial.read(); //reads the data into a variable
    Serial.println(input);
    delay(15);

    switch (input){
    case 'd': value2 = Serial.parseInt();
        Serial.println(value2);
        Serial.println("going down ...");//down
        p_y = servoY.read();
        if (p_y >= 30){
           p_y -= value2; 
           servoY.write(p_y);
           delay(10);} 
        break;   
        
    case 'u': value2 = Serial.parseInt();
         Serial.println(value2);
         Serial.println("going up ...");//down
         p_y = servoY.read();
         if (p_y <=103.5 ){
           p_y += value2;
           servoY.write(p_y); 
           delay(10);}    //adjusts the servo angle according to the input
         break;
         
     case 'l': value2 = Serial.parseInt();
          Serial.println(value2);
          Serial.println("going left...");
          p_x = servoX.read();
          if (p_x >= 0){
            p_x += value2;               
            servoX.write(p_x); 
            delay(10);}
          break;

     case 'r': value2 = Serial.parseInt();
         Serial.println(value2);
         Serial.println("going right...");
         p_x = servoX.read();
         if (p_x <= 200){
            p_x -= value2;               
            servoX.write(p_x);
            delay(10);}
          break;


      // to find the ball when its out of sight
      case 'D':
        Serial.println("finding ball down ...");//down
        p_y = servoY.read();
        if (p_y >=30){
           p_y -= 1;               
           servoY.write(p_y);
           delay(10);
        } 
        break;  
      
         
      case 'U':
          Serial.println("finding ball up ..."); //up
          p_y = servoY.read();
          if (p_y <=103.5){
             p_y += 1;               
             servoY.write(p_y); 
             delay(10);}
          break;
          
      case 'L':
          Serial.println("finding ball on the left...");
          p_x = servoX.read();
          if (p_x >= 0){
            p_x += 1;               
            servoX.write(p_x); 
            delay(10);}
          break;

      case 'R':
         Serial.println("finding ball on the right...");
         p_x = servoX.read();
         if (p_x <= 200){
            p_x -= 1;               
            servoX.write(p_x); 
            delay(10);}
         break;
     
          
     default:
        Serial.println("Executing default function ...");
        p_x = servoX.read();
        p_y = servoY.read();
        setup_servos(p_x, p_y);
        Serial.println("Done.");
        delay(10);
        break; 
    }
  }
}
