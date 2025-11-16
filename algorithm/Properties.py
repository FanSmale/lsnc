import os
import json
class Properties:
    """
    The algorithm parameters.
    """

    def __init__(self, mat_name: str = 'default', scheme: int = 0):

        # check the JSON file
        config_file_name = '../config/config.json'
        assert os.path.exists(config_file_name), 'Config file is not accessible.'
        # open file
        with open(config_file_name) as f:
            cfg = json.load(f)['masp']
        self.name = mat_name
        self.cross_val = cfg['common']['crossValidation']
        self.cross_validate_num = cfg['common']['crossNum']

        self.learning_rate = cfg['common']['learningRate']
        self.budget = cfg['common']['budget']
        self.train_data_matrix = cfg['common']['trainDataMatrix']
        self.test_data_matrix = cfg['common']['testDataMatrix']
        self.train_label_matrix = cfg['common']['trainLabelMatrix']
        self.test_label_matrix = cfg['common']['testLabelMatrix']

        self.increment_rounds = cfg['common']['incrementRounds']
        # threshold
        self.enhancement_threshold = cfg['common']['enhancementThreshold']
        self.num_instances = cfg['common']['numInstances']
        self.num_conditions = cfg['common']['numConditions']

        self.num_labels = cfg['common']['numLabels']

        self.learning_rate = cfg['common']['learningRate']
        self.mobp = cfg['common']['mobp']

        assert mat_name in cfg.keys(), "".join(
            ['The parameters of ', mat_name, 'are not defined in the JSON file of config.'])
        temp_dataset_cfg = cfg[mat_name]
        self.filename = temp_dataset_cfg['fileName']
        print(self.filename)
        self.output_filename = temp_dataset_cfg['outputFileName']

        assert os.path.exists(self.filename), 'Dataset file is not accessible.'

        self.parallel_layer_num_nodes = temp_dataset_cfg['parallelLayerNumNodes']

        self.k_label = temp_dataset_cfg['kLabel']

        self.parallel_layer_num_nodes[-1]=2 ** self.k_label
        self.outputFileName = temp_dataset_cfg['outputFileName']

        if scheme == 1:
            temp_array = [0, 2]
            temp_array[0] = self.parallel_layer_num_nodes[0]
            self.parallel_layer_num_nodes = temp_array

        self.parallel_activators = temp_dataset_cfg['parallel_activators']
        self.full_connect_activators = temp_dataset_cfg['full_connect_activators']

        self.full_connect_layer_num_nodes = temp_dataset_cfg['fullConnectLayerNumNodes']

        if scheme == 2:
            temp_array2 = [0, 0]
            temp_array2[-1] = self.full_connect_layer_num_nodes[-1]
            self.full_connect_layer_num_nodes = temp_array2
