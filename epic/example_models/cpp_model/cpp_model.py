import cpp_model

from epic.core.model import Model


class CppModel(Model):
    def forward(self, param):
        return cpp_model.forward(param)

    def jacobian(self, param):
        return cpp_model.forward(param)
