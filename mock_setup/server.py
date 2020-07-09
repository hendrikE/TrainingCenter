import sys
sys.path.insert(1, '/home/hendrik/Bachelorthesis/TrainingCenter/analysis')

import functions
import pickle
import numpy as np
import paho.mqtt.client as mqtt

data = []
size = [5, 5, 5]


def on_message_data(client, userdata, message):
    info = message.payload.decode("utf-8")
    info = info.replace("[", "")
    info = info.replace("]", "")
    info = info.split(",")
    info = [float(x) for x in info]
    data.append(info)
    print("Added message '{}' to the data set".format(message.payload.decode("utf-8")))


def on_message_state(client, userdata, message):
    client.disconnect()


def run():
    with open("model", "rb") as file:
        model = pickle.load(file)
    client = mqtt.Client()

    while True:
        message = input("Start a new run [y/n]: ")
        data.clear()
        if message == "y":
            client.connect("localhost")
            client.publish("state", "1")
            print("Send state message")
            client.subscribe("data")
            client.subscribe("state")
            client.message_callback_add("state", on_message_state)
            client.message_callback_add("data", on_message_data)
            print("Subscribed to data and state, start loop")
            client.loop_forever()
            info = np.array(data)
            grid = functions.turn_coordinates_into_grid(info[:, :3], info[:, 6], size)
            features = functions.turn_grid_into_features(grid, size)
            result = model.predict([features])
            print(result)
        else:
            break


if __name__ == "__main__":
    run()
