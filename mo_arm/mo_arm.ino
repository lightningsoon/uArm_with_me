#include <CurieBLE.h>
#include <Servo.h>


#define CRAW_MIN 0 //open 
#define CRAW_MAX 40 //close
#define ELBOW_DEFAULT 75 //前臂
#define SHOULDER_DEFAULT 80 //后臂
#define WRIST_X_DEFAULT 90
//#define WRIST_Y_DEFAULT 90
#define WRIST_Z_DEFAULT 90
#define BASE_DEFAULT 5 //底座
#define CRAW_DEFAULT CRAW_MIN
#define DELAY_TIME 800

Servo myservoA;
Servo myservoB;
Servo myservoC;
//Servo myservoD;
Servo myservoE;
Servo myservoF;
Servo myservoG;//爪子

String comdata = "";
//定义一个comdata字符串变量，赋初值为空值
const int ori_numdata[5] = {0};
int numdata[5] = {0};
int mark = 0;
//numdata是分拆之后的数字数组
void sleep();
void reset();
void setup() {

  Serial.begin(9600);

  myservoA.attach(2);
  myservoB.attach(3);
  myservoC.attach(4);
  //myservoD.attach(5);
  myservoE.attach(6);
  myservoF.attach(7);
  myservoG.attach(8);

  reset();
}

void loop() {
  int j = 0;
  //j是分拆之后数字数组的位置记数

  //不断循环检测串口缓存，一个个读入字符串，
  if (Serial.available() > 0)
  {
    while (Serial.available() > 0)
    {
      //读入之后将字符串，串接到comdata上面。
      comdata += char(Serial.read());
      //延时一会，让串口缓存准备好下一个数字，不延时会导致数据丢失，
      delay(2);
    }
    //接收到数据则执行comdata分析操作，否则什么都不做。
    {
      //显示刚才输入的字符串（可选语句）
//      Serial.println(comdata);
      //显示刚才输入的字符串长度（可选语句）
      // Serial.println(comdata.length());

      /*******************下面是重点*******************/
      //以串口读取字符串长度循环，
      int leng = int(comdata.length());

      for (int i = 0; i < leng ; i++)
      {
        //逐个分析comdata[i]字符串的文字，如果碰到文字是分隔符（这里选择逗号分割）则将结果数组位置下移一位
        //即比如11,22,33,55开始的11记到numdata[0];碰到逗号就j等于1了，
        //再转换就转换到numdata[1];再碰到逗号就记到numdata[2];以此类推，直到字符串结束
        if (comdata[i] == ',')
        {
          j++;
        }
        else
        {
          numdata[j] = numdata[j] * 10 + (comdata[i] - '0');
        }
      }
      comdata = String("");

      //开爪
      //       delay(DELAY_TIME);
      //       myservoG.write(CRAW_MIN);   //初始值为58，爪子夹紧，度数越小爪口越小

      //移动到抓取目标
      delay(DELAY_TIME);
      myservoB.write(120);//收拳
      delay(DELAY_TIME);
      myservoA.write(numdata[2]);      //初始值为60，机械前臂伸缩，度数减少向前伸缩
      
      delay(DELAY_TIME);
      myservoF.write(numdata[0]);  //初始值为60，度数<90逆时针旋转，度数>90顺时针旋转

      sleep();
      myservoB.write(numdata[1]);      //初始值为90，机械后臂伸缩，度数减少向前伸缩
      
      //      delay(DELAY_TIME);
      //       myservoC.write(numdata[3]);
      // delay(DELAY_TIME);
      // myservoE.write(numdata[4]); //初始值为90，爪子旋转，度数增大顺时针，度数减少逆时针

      //抓
        delay(DELAY_TIME);
        myservoG.write(CRAW_MAX);   //初始值为58，爪子夹紧，度数越大爪口越小

      //倒
      //      sleep();
      //      myservoF.write(180);
      //      sleep();
      //  myservoC.write(WRIST_X_DEFAULT);
      //   //delay(DELAY_TIME);
      // // myservoD.write(WRIST_Y_DEFAULT);
      //  delay(DELAY_TIME);
      //  myservoE.write(WRIST_Z_DEFAULT); //初始值为90，爪子旋转，度数增大顺时针，度数减少逆时针
      //   delay(DELAY_TIME);
      //  myservoF.write(BASE_DEFAULT);

      //放
        sleep();
        myservoF.write(260);
        
        sleep();
        myservoE.write(180);
        delay(1000);
        myservoE.write(WRIST_Z_DEFAULT);
        delay(1000);
        myservoG.write(CRAW_MIN);


      //   //初始化
      reset();
      Serial.println(1);
    }
  }
}
void sleep()
{
  delay(DELAY_TIME);
}

void reset()
{
  sleep();
  myservoF.write(BASE_DEFAULT); //底座
  sleep();
  myservoB.write(SHOULDER_DEFAULT);//后
  sleep();
  myservoA.write(ELBOW_DEFAULT);//前
  sleep();
  myservoF.write(BASE_DEFAULT); //底座
  sleep();
  myservoC.write(WRIST_X_DEFAULT);
  //// myservoD.write(WRIST_Y_DEFAULT);
  myservoE.write(WRIST_Z_DEFAULT);//爪子旋转
  
  myservoG.write(CRAW_DEFAULT);    //爪子
  for (int i = 0; i < 5; i++)
  {
    numdata[i] = ori_numdata[i];
  }
}

