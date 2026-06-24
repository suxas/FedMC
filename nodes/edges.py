class EdgeSBS:
    def __init__(self, sbs_id, config):
        self.id = sbs_id
        self.config = config
        # SBS 通用的物理属性可在此扩展，如最大带宽、计算能力等