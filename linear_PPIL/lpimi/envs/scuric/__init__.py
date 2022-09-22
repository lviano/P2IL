"""Python Script Template."""
from gym.envs.registration import register

base = "lpimi.envs.scuric"
register(id="LeftChain-v0", entry_point=f"{base}.left_chain:LeftChain")
register(id="TwoStateStochastic-v0", entry_point=f"{base}.two_state:TwoState")
register(id="WindyGrid-v0", entry_point=f"{base}.windy_grid:WindyGrid")
register(id="DeepSea-v0", entry_point=f"{base}.deep_sea:DeepSea")
register(id="WideTree-v0", entry_point=f"{base}.wide_tree:WideTree")
register(id="RiverSwim-v0", entry_point=f"{base}.river_swim:RiverSwim")
