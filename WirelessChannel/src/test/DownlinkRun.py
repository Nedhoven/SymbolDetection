
import src.main.Downlink as Dl
import src.main.Precoder as Pr


def start():
    """setting a sample downlink"""
    antenna = 512
    user = 10
    block = 100
    length = 4
    pre = Pr.MatchedFilter(antenna_count=antenna, user_count=user, channel_length=length)
    downlink = Dl.DownlinkGenerator(antenna_count=antenna, user_count=user, channel_length=length, block_length=block)
    return
