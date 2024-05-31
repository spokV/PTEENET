args_global = \
    [
        '--dataset', 'cifar10',
        '--model', 'eenet110',
        '--num-ee', '10',
        '--filters', '4',
        '--exit-type', 'conv2',
        '--distribution', 'fine',
        #'--add-noise',
        '--min-noise-snr', '-5',
        '--max-noise-snr', '30',
        '--num-noise-levels', '5',
        '--validate',
    ]

argu = [

#args_train_main = \
    [ 
        '--epochs', '500',
        '--optimizer', 'Adam',
        '--lr', '0.001',
        '--weight-decay', '0.0005',
        #'--adaptive-lr',
        '--batch-size', '128',
        '--test-batch', '128',
        '--momentum', '0.9',
        #'--testing',
        '--no-tensorboard',
        '--clear-dirs',
        '--shuffle-train',
        '--log-interval', '1',
        '--save-best',
        '--early-stopping',
        '--loss-func', 'v6'
        # '--plot-history',
        # '--no-save-model'
    ],

#args_train_ee = \
    [ 
        '--epochs', '30',
        '--lambda-coef', '0.9',
        '--optimizer', 'Adam',
        '--lr', '0.0005',
        #'--weight-decay', '0.0001',
        '--load-model', 'main_models/models/cifar10/eenet110/after_main_training/ee10/conv2/model.pt',
        #'--load-model', 'main_models/models/cifar10/eenet110/after_main_training/ee10/conv/model.pt',
        #'--load-model', 'main_models/models/cifar10/eenet110/after_main_training/ee10/conv2_bnpool/model.pt',
        #'--load-model', 'main_models/models/cifar10/eenet32/after_main_training/ee5/model.pt',
        #'--use-main-targets',
        #'--early-stopping',
        # '--plot-history',
        '--loss-func', 'v2',
        #'--no-tensorboard',
        '--clear-dirs',
        '--batch-size', '128',
        '--test-batch', '128',
        '--log-interval', '1',
        '--loss-threshold', '0.5',
        '--shuffle-train',
        '--termination', 'confidence',
        #'--no-save-model',
        #'--ee-costs', '0.1282048606925034', '0.27', '0.38'
    ],

#args_generate_exits_ground_truths = \
    [ 
        '--lambda-coef', '0.0',
        '--load-model', 'main_models/models/cifar10/eenet110/UT/clean/after_ee_training/model.pt',
        '--use-main-targets',
        '--no-tensorboard',
        #'--shuffle-train',
        #'--shuffle-test',
        '--batch-size', '128',
        '--test-batch', '128'
    ],

#args_train_gating_model = \
    [ 
        '--epochs', '300',
        '--start-epoch', '0',
        '--optimizer', 'SGD',
        '--lr', '0.001',
        '--weight-decay', '0.0001',
        '--momentum', '0.9',
        #'--no-tensorboard',
        '--loss-threshold', '0.2',
        '--batch-size', '128',
        '--test-batch', '128',
        '--max-depth', '10',
    ],

#args_generate_relative_loss = \
    [ 
        #'--epochs', '30',
        #'--lr', '0.0001',
        #'--momentum', '0.9',
        '--test-batch', '200',
        '--loss-threshold', '1.1',
        '--load-model', 'main_models/models/cifar10/eenet110/UT/clean/after_ee_training/model.pt',
        '--loss-main', '0.22',
        '--max-depth', '4',
    ],

#args_plot_relative_loss = \
    [
    ],

#args_test_gated = \
    [
        '--test-batch', '128',
        '--loss-threshold', '0.2',
        '--load-model', 'main_models/models/cifar10/eenet110/UT/clean/after_ee_training/model.pt',
        #'--max-depth', '10',
        '--use-main-targets',
    ]
]
