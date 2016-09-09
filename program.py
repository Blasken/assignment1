import ex1
import ex2
import common

if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) > 0:
        pass

    train_data = common.read_data('train_data_2016.txt')
    vailid_data = common.read_data('valid_data_2016.txt')

    result1 = ex1.run()
