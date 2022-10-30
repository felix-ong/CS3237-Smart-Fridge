#include <ArduinoMqttClient.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include "secret.h"
#define PHOTORESISTOR A0

const char* ssid = SECRET_SSID; //Your Wifi's SSID
const char* password = SECRET_PASS; //Wifi Password

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

const char* broker = "172.25.104.209";
int port = 1883;
const char* topic = "fridge/photoresistor";

//set interval for sending messages (milliseconds)
const long interval = 500;
unsigned long previousMillis = 0;

int count = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  WiFi.begin(ssid, password);
  Serial.println("");

  Serial.print("Attempting to connect to WPA SSID: ");
  Serial.println(ssid);

  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  pinMode(PHOTORESISTOR, INPUT); 
  Serial.print("Attempting to connect to the MQTT broker: ");
  Serial.println(broker);

  if (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());
    while (1);
  }
  Serial.println("You're connected to the MQTT broker!");
  Serial.println();
}

void loop() {
  // put your main code here, to run repeatedly:
  
  // call poll() regularly to allow the library to send MQTT keep alive which
  // avoids being disconnected by the broker
  mqttClient.poll();

  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    // save the last time a message was sent
    previousMillis = currentMillis;
    
    int reading = analogRead(A0);
    
    Serial.print("Sending message to topic: ");
    Serial.println(topic);
    Serial.println(reading);

    // send message, the Print interface can be used to set the message contents
    mqttClient.beginMessage(topic);
    mqttClient.print(reading);
    mqttClient.endMessage();
    Serial.println();
  }
}
