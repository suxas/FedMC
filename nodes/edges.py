class EdgeSBS:
    """边缘小基站(SBS)：负责管理其覆盖范围内的客户端"""

    def __init__(self, sbs_id, config):
        self.id = sbs_id
        self.config = config