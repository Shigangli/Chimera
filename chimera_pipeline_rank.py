
class AutoGeneratePipelineRank:

    def __init__(self, stage_numbers, divisors, micro_batch_numbers):
        self.module_to_stage_map = [i for i in range(stage_numbers)]
        self.stage_numbers = stage_numbers
        assert divisors % 2 == 0, "pipeline num must be an even number"
        self.pipeline_numbers = 1 if divisors is None else divisors//2
        self.micro_batch_numbers = micro_batch_numbers
        self.push_pipeline_numbers = {
            "up": 0,
            "down": 0
        }
        self.push_micro_batch = 0

    def generate_pipeline(self):
        self.up_pipline_list = []
        self.down_pipeline_list = []
        for i in range(self.pipeline_numbers):
            # generate up pipeline
            micro_num = self.stage_numbers//(2*self.pipeline_numbers)
            if self.micro_batch_numbers-self.push_micro_batch <= micro_num:
                micro_num = self.micro_batch_numbers-self.push_micro_batch

            self.push_micro_batch += micro_num

            self.up_pipline_list.append(MyPipeLine(i,
                                                   micro_num, self.stage_numbers,
                                                   self.pipeline_numbers, self.module_to_stage_map, True))

            if self.micro_batch_numbers-self.push_micro_batch <= micro_num:
                micro_num = self.micro_batch_numbers-self.push_micro_batch

            self.push_micro_batch += micro_num

            # generate down pipeline
            self.down_pipeline_list.append(MyPipeLine(i,
                                                      micro_num, self.stage_numbers,
                                                      self.pipeline_numbers, self.module_to_stage_map, False))

    def get_schedule(self, is_iteration=False):
        schedule = []
        schedule_up_down = []
        pipelines = self.up_pipline_list + self.down_pipeline_list

        for i in range(self.stage_numbers):
            schedule.append(list())
            schedule_up_down.append(list())
        has_next_flag = True
        has_next_sync = 0
        steps = 0
        sync_list = [[] for i in range(self.stage_numbers)]
        while(has_next_flag or has_next_sync != 0):
            next_flag = False
            sub_schedule = list("" for _ in range(self.stage_numbers))
            for index, pipeline in enumerate(pipelines):
                if pipeline.has_next_pass():
                    next_data, is_pop, step_direction, up_or_down, is_sync = pipeline.next_pass()

                    for k in next_data.keys():
                        schedule[next_data[k] %
                                 self.stage_numbers].append(str(k))
                        up_or_down_str = str(index)
                        up_or_down_str += "@down@" if up_or_down else "@up@"
                        schedule_up_down[next_data[k] %
                                         self.stage_numbers].append(f"{up_or_down_str}{'f' if step_direction[k] == 1 else 'b'}")
                        if step_direction.get(pipeline.micro_batch_ids[-1], 1) != 1:
                            direction = "down"
                            if pipeline.up_or_down:
                                direction = "up"

                        sub_schedule[next_data[k] %
                                     self.stage_numbers] = f"{up_or_down_str}{'f' if step_direction[k] == 1 else 'b'}"
                    if is_sync and next_data.get(pipeline.micro_batch_ids[-1]) is not None:
                        has_next_sync += 1
                        sync_list[next_data[pipeline.micro_batch_ids[-1]] %
                                  self.stage_numbers].append(f"{up_or_down_str}s")
                    if is_pop and pipeline.has_next_pass():
                        micro_num = self.stage_numbers//(2 *
                                                         self.pipeline_numbers)
                        if self.micro_batch_numbers-self.push_micro_batch <= micro_num:
                            micro_num = self.micro_batch_numbers-self.push_micro_batch

                        self.push_micro_batch += micro_num
                        if micro_num != 0:
                            if pipeline.up_or_down:
                                direction = "up"
                            else:
                                direction = "down"
                            pipelines.append(MyPipeLine(self.pipeline_numbers+self.push_pipeline_numbers[direction],
                                                        micro_num, self.stage_numbers,
                                                        self.pipeline_numbers, self.module_to_stage_map, pipeline.up_or_down))
                            self.push_pipeline_numbers[direction] += 1

                    next_flag = True
            for index, s in enumerate(sub_schedule):
                if s == "" and len(sync_list[index]) > 0:
                    sub_schedule[index] = sync_list[index].pop(0)
                    has_next_sync -= 1

            for i in range(self.stage_numbers):
                if len(schedule[i]) <= steps:
                    schedule[i].append(sub_schedule[i])
                    schedule_up_down[i].append(sub_schedule[i])

            steps += 1
            has_next_flag = next_flag
            if is_iteration and has_next_flag:
                yield sub_schedule

class MyPipeLine:
    def __init__(self, pipeline_id, micro_batch_numbers,
                 stage_numbers, pipeline_numbers, module_to_stage_map, up_or_down):

        self.pipeline_id = pipeline_id
        self.micro_batch_numbers = micro_batch_numbers
        self.stage_to_rank_map = None
        self.pipeline_numbers = pipeline_numbers
        self.stage_numbers = stage_numbers
        self.module_to_stage_map = module_to_stage_map
        self.up_or_down = up_or_down
        self.devices = None


        self.steps = -1
        self.step_direction = dict()
        self.micro_batch_ids = list()
        self.micro_batch_device = dict()
        micro_batch_id = ((self.pipeline_id//2)*self.stage_numbers)
        micro_batch_id += (0 if self.up_or_down else self.stage_numbers//2)
        for x in range(self.micro_batch_numbers):
            self.micro_batch_ids.append(
                x+(self.pipeline_id % self.pipeline_numbers)*(self.stage_numbers//self.pipeline_numbers//2)+micro_batch_id)

        start_stage_device = (self.pipeline_id % self.pipeline_numbers) * \
            (self.stage_numbers // self.pipeline_numbers)
        self.devices = [x for x in self.module_to_stage_map[start_stage_device:] +
                        self.module_to_stage_map[:start_stage_device]]

        if self.up_or_down is True:
            # down pipeline
            self.stage_to_rank_map = {
                str(index): [device] for index, device in enumerate(self.devices)}
        else:
            # up pipeline
            self.stage_to_rank_map = {
                str(self.stage_numbers-1-index): [device] for index, device in enumerate(self.devices)}

    def next_pass(self):
        if self.steps <= (self.micro_batch_numbers-1) * 2:
            self.steps += 1

        over_back_micro_batch = []
        for micro_batch in self.micro_batch_device.keys():
            step = 1 if self.up_or_down else -1
            if self.step_direction[micro_batch] == 1 and abs(self.micro_batch_device[micro_batch] - (self.stage_to_rank_map["0"][0] + 2*self.stage_numbers)) >= self.stage_numbers-1:
                self.step_direction[micro_batch] = -1
            elif self.step_direction[micro_batch] == -1 and self.micro_batch_device[micro_batch] == self.stage_to_rank_map["0"][0] + 2*self.stage_numbers:
                over_back_micro_batch.append(micro_batch)
            else:
                self.micro_batch_device[micro_batch] += step * \
                    self.step_direction[micro_batch]
        pop_one = False
        for micro_batch in over_back_micro_batch:
            self.micro_batch_device.pop(micro_batch)
            pop_one = True

        if self.steps % 2 == 0:
            self.micro_batch_device[self.micro_batch_ids[self.steps //
                                                         2]] = self.stage_to_rank_map["0"][0] + 2*self.stage_numbers
            self.step_direction[self.micro_batch_ids[self.steps // 2]] = 1
        is_sync = True if self.step_direction.get(
            self.micro_batch_ids[-1]) == -1 else False
        return self.micro_batch_device, pop_one, self.step_direction, self.up_or_down, is_sync

    def has_next_pass(self):
        if self.micro_batch_numbers > 0 and (self.steps == -1 or self.micro_batch_device):
            return True
        return False


# ##########################测试用
def forward(up_or_down):
    print(f"forward:{up_or_down}")


def backward(up_or_down):
    print(f"backward:{up_or_down}")


if __name__ == "__main__":
    stage_num = 8
    pipeline_num = 2
    micro_num = 8
    print(
        f"stage:{stage_num}  pipeline_num:{pipeline_num} micro_num:{micro_num}")
    pipeline = AutoGeneratePipelineRank(stage_num, pipeline_num, micro_num)
    pipeline.generate_pipeline()
    stage_to_ranks = [[] for i in range(stage_num)]
    for pipe in pipeline.down_pipeline_list:
        for k, v in pipe.stage_to_rank_map.items():
            stage_to_ranks[int(k)].append(*v)
    for pipe in pipeline.up_pipline_list:
        for k, v in pipe.stage_to_rank_map.items():
            stage_to_ranks[int(k)].append(*v)
    s = pipeline.get_schedule(True)
    for sub_schedule in s:
        if sub_schedule[0] != '':
            up_down, forward_backward = sub_schedule[0].split(
                "@")
            if forward_backward == 'f':
                forward(up_down)
            else:
                backward(up_down)
        else:
            print("")
    pass
