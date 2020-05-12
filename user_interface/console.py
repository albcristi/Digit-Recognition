from utils.utils import *
from cnn_model.cnn_model import *

class Console:


    def printMenu(self):
        print("1 - Run CNN")
        print("x - Exit")

    def getCommand(self):
        return input("Enter command:\n>>> ")

    def runConsole(self):
        while True:
            try:
                self.printMenu()
                command = self.getCommand()
                if command not in ["1", "x"]:
                    raise ValueError("Invalid command!")
                if command == "x":
                    print("Good Bye! :D")
                    break
                else:
                    self.__runCommand1()
            except Exception as er:
                print(er)

    def __runCommand1(self):
        print("Our CNN will run 3 epochs")
        print("A plot will be shown in order\nto see the performance" +
              "\nYou will also see some predictions")

        # load data
        x_train, y_train, x_test, y_test = load_data_set()

        # reshape data
        x_train = reshape_data_set(x_train, 60000)
        x_test = reshape_data_set(x_test, 10000)

        # encoding
        y_train_one_hot = hot_encoding(y_train)
        y_test_one_hot = hot_encoding(y_test)

        # create and train model
        cnn = CnnModel()
        cnn.train_model(x_train, y_train_one_hot, x_test, y_test_one_hot)

        # show plot
        cnn.plot_results()

        # get some predictions
        predictions = cnn.predict(x_test[:20])
        for index in range(0,20):
            print("Digit: "+str(y_test[index])+" VS predicted digit: "+str(predictions[index]))
