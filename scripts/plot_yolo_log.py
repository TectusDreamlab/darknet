import argparse
import sys
import matplotlib.pyplot as plt

def main(argv):

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "-log_file",
        required = True,
        help = "path to log file")

    args = parser.parse_args()

    f = open(args.log_file)

    lines  = [line.rstrip("\n") for line in f.readlines()]

    numbers = {'1','2','3','4','5','6','7','8','9'}

    iters = []
    loss = []
    learning_rate = []
    time_used = []

    for line in lines:
        args = line.split(' ')

        try:
            if args[0][-1:]==':' and args[0][0] in numbers :
                iters.append(int(args[0][:-1]))
                loss.append(float(args[2]))
                learning_rate.append(float(args[4]))
                time_used.append(float(args[6]))
        except Exception as inst:
            print inst

    # plot with various axes scales
    plt.figure(1)

    # loss
    plt.subplot(311)
    plt.plot(iters, loss)
    plt.xlabel('iterations')
    plt.ylabel('average loss')
    plt.yscale('linear')
    plt.grid(True)


    # learning rate
    plt.subplot(312)
    plt.plot(iters, learning_rate)
    plt.xlabel('iterations')
    plt.ylabel('learning rate')
    plt.yscale('linear')
    plt.grid(True)

    # time consumed
    plt.subplot(313)
    plt.plot(iters, time_used)
    plt.xlabel('iterations')
    plt.ylabel('seconds')
    plt.yscale('linear')
    plt.grid(True)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.2, right=0.95, hspace=0.5,
                    wspace=0.35)
    plt.show()
if __name__ == "__main__":
    main(sys.argv)