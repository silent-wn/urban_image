class opt:
    learningRate = 0.002             # learning rate  
    beta1 = 0.9                      # momentum term for adam
    batchSize = 100               # batch size  
    gpu = 2    
    trainFrac = 0.7           # gpu to use  
    save = '/media/USERDATA/xrj/01architectures/exp1'          # base directory to save logs  
    classNum = 5
    dataRoot = "/media/DATASET/project/01architectures" # data root directory
    eval = True           # optimizer to train with
    nEpochs  = 200              # max training epochs  
    seed      = 1                # random seed  