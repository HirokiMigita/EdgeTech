#define IN1 13 //motor1
#define IN2 12 //motor1
#define IN3 11 //motor2
#define IN4 10 //motor2
#define IN5 9  //sorenoid

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(IN5, OUTPUT);
  Serial.begin(9600);
}

void AirSupply(int time) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  //digitalWrite(IN3, LOW);
  //digitalWrite(IN4, LOW);
  digitalWrite(IN5, HIGH);
  delay(time);
}

void AirExhaust(int time) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  //digitalWrite(IN3, HIGH);
  //digitalWrite(IN4, LOW);
  digitalWrite(IN5, LOW);
  delay(time);
}

void AirStop(int time) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  //digitalWrite(IN3, LOW);
  //digitalWrite(IN4, LOW);
  digitalWrite(IN5, HIGH);
  delay(time);
}

// シリアル通信で受信したデータを数値に変換
void serialNumVal(){
  // データ受信した場合の処理
  if (Serial.available()>0) {
    int input = Serial.parseInt();
    Serial.println(input);

    if(input == 1){
      AirSupply(6000);
      AirExhaust(500);
      AirSupply(1500);
      AirExhaust(500);
      AirSupply(1500);
      AirExhaust(500);
      AirSupply(1500);
      AirExhaust(15000);
    }else{
      return ;
    }
  }
}

void loop() {
  serialNumVal();
}
