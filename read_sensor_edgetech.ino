#define INTERVAL_TIME 5  /**< 周期実行の間隔 */
unsigned long prev_time;     /**< 周期実行の前回実行時刻 */
unsigned long interval_time; /**< 周期実行の間隔 */
#define BUF_SIZE 50       /**< シリアル通信の送信バッファのサイズ */
#define AIN_NUM 4 /** アナログ入力ピンの数 */

void sendData() {
  uint16_t i;
  uint16_t ain_values[AIN_NUM];
  char buf[BUF_SIZE];
  
  for (int i = 0; i < AIN_NUM; i++) {
    ain_values[i] = analogRead(i);
  }

  sprintf(
    buf,
    "%1u,%1u,%1u,%1u",
    ain_values[0],
    ain_values[1],
    ain_values[2],
    ain_values[3]);
  Serial.println(buf);
  
}

void setup() {
  uint16_t i;
  // put your setup code here, to run once:
  Serial.begin(115200);
  //アナログ入力ピンの初期化
  for(i = 0; i < AIN_NUM; i++){
    pinMode(i,INPUT_PULLUP);
  }
  // 周期実行の初期化
  prev_time = 0;
  interval_time = INTERVAL_TIME;
}

void loop() {
  // put your main code here, to run repeatedly:
  unsigned long curr_time;

  // 現在時刻を取得する
  curr_time = millis();

  // 周期実行
  if ((curr_time - prev_time) >= interval_time) {
    prev_time += interval_time;
    sendData();
  }
}
