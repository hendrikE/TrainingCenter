import time
import numpy as np
import paho.mqtt.client as mqtt
from scipy.stats import multivariate_normal


def on_message_state(client, userdata, message):
    client.disconnect()
    print("Switched to State 1")


def run():
    distribution = np.load("1.npy")
    distribution = distribution[-1]
    distribution = multivariate_normal(distribution[:3], np.diag(distribution[3:]))
    segmentation = np.load("5_5_5.npy")
    sample = distribution.pdf(segmentation[:, 3:])
    print("Loaded data")
    client = mqtt.Client()
    client.connect("localhost")
    while True:
        client.subscribe("state")
        client.on_message = on_message_state
        print("Subscribed to state topic")
        print("Starting to wait for state message")
        client.loop_forever()
        client.connect("localhost")
        print("Reconnected")
        for it, value in enumerate(sample):
            client.publish("data", str(segmentation[it].tolist() + [value]))
            print("Sent data: '{}'".format(str(segmentation[it].tolist() + [value])))
            time.sleep(1)
        client.publish("state", "2")


if __name__ == "__main__":
    run()
