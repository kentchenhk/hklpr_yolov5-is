from lpr_mtr import lpr_mtr

if __name__ == '__main__':
    LPR_MTR = lpr_mtr(yaml_path='./lpr_mtr.yaml')
    LPR_MTR.run() # *Call LPR_MTR.stop() when you want to stop the thread.