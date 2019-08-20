class GeneralUtils():

    @classmethod
    def is_classification(cls, model):
        return hasattr(model, 'predict_proba')

    @classmethod
    def dmd_supported(cls, model, dmd):
        try:
            model.predict(dmd.split_by_indices(indices=[0, 1, 2]))
            return True
        except:
            return False
