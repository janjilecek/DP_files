# author: Jan Jilecek

DEBUG = False


class Challenge:
    def __init__(self, _key, _name, _stages_duration, _stages_data, _text):
        self.key = _key
        self.name = _name
        self.stages_duration = _stages_duration
        if DEBUG:
            self.stages_duration = [1,1,1]
        self.stages_data = _stages_data
        self.text = _text
        self.original_duration = _stages_duration

    def get_duration(self, stage):
        return self.stages_duration[stage - 1]

    def get_original_duration(self, stage):
        return self.stages_duration[stage - 1]  # TODO decorator function

    def get_data(self, stage):
        if stage == 2 or stage == 3:
            print("get_data " + str(stage) + "," + str(self.stages_data[stage - 2]))
            return self.stages_data[stage - 2]
        else:
            # raise Exception("Stage Data out of bounds")
            print("Stage Data out of bounds")
            return self.text

    def get_name(self):
        return self.name

    def get_text(self):
        return self.text

    def get_key(self):
        return self.key
