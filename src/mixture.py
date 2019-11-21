from expert import Expert
from sparse_gate import SparseGate

class Mixture():
    def __init__(self, x_train, y_train, x_test, y_test, experts):
        self.gate = SparseGate(x_train, y_train, x_test, y_test)
        self.gate.create_gate_model(experts)

        self.experts = experts
        
    def load_expert_weights_and_set_trainable_layers(self,weights_file='lib/weights/base_model_'):
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

    def train(self, datagen, weights_file_out):
        self.load_expert_weights_and_set_trainable_layers()
        self.gate.train_gate(datagen, weights_file_out)

    def add_expert(self, datagen, weights_file_in, model_prev, expert):
        self.experts.append(expert)
        self.gate.load_gate_weights(self.gate.model, model_prev)
        self.load_expert_weights_and_set_trainable_layers()
        self.gate.train_gate(self.gate,weights_file_in)