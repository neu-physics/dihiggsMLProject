import numpy as np
import uproot, uproot_methods
class JetCons:
    def __init__(self, UID=-999, PT=-999, Eta=-999, Phi=-999, E=-999, **kwarg):
        #self.UID = UID
        #self.PT = PT
        #self.Eta = Eta
        #self.Phi = Phi
        #self.E = E
        self.cons_LVec = uproot_methods.TLorentzVector.from_ptetaphi(PT,Eta,Phi,E)
        #self.y = self.cons_LVec.rapidity
        self.property = {'UID': UID,
                         'PT': PT,
                         'Eta': Eta,
                         'Phi': Phi,
                         'E': E,
                         'rapidity': self.cons_LVec.rapidity,
                         **kwarg
                        }

    def get(self, _name):
        if(_name in self.property.keys()):
            return self.property[_name]
        else:
            print("It is not available...")
        return

    
