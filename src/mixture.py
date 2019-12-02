from expert import Expert
from sparse_gate import SparseGate

class Mixture():
    def __init__(self, x_train, y_train, x_test, y_test, experts, inputs, spark_context):
        self.gate = SparseGate(x_train, y_train, x_test, y_test, inputs, spark_context)
        self.experts = experts
        self.model_previous = None
        
    def load_expert_weights_and_set_trainable_layers(self,weights_file='../lib/weights/base_model_'):
        model = self.gate.gateModel
        for a in range(len(self.experts)):
            m = self.experts[a]
            file = weights_file + str(a) + '.h5'
            m.load_weights(file, by_name=True)
            for b in m.layers:
                for l in model.layers:
                    if (l.name == b.name):
                        l.set_weights(b.get_weights())
                        print("loaded layer "+str(l.name))

        for l in model.layers:
            if ('gate' in l.name or 'lambda' in l.name):
                l.trainable = True
                # print("training gate ")
            else:
                l.trainable = False

    def train_init(self, datagen, weights_file):
        for i in range(1, len(self.experts)):
            print("---------------------------------------")
            print(str(i) + " of " + str(len(self.experts)))
            print("---------------------------------------")
            
            self.gate.gateModel = self.gate.create_gate_model(self.experts[:i])
            if i > 1:
                self.gate.load_gate_weights(self.model_previous)
            self.load_expert_weights_and_set_trainable_layers()
            self.gate.train_gate(datagen, weights_file) 
            self.model_previous = self.gate.gateModel

    def add_expert(self, datagen, weights_file, expert):
        self.experts.append(expert)
        self.gate.load_gate_weights(self.gate.model, self.model_previous)
        self.load_expert_weights_and_set_trainable_layers()
        self.gate.train_gate(self.gate,weights_file)
        self.model_previous = self.gate.gateModel