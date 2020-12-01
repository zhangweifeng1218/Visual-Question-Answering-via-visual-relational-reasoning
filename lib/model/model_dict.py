from lib.model import reasoning_vkg
from lib.model import bottom_up
from lib.model import relationalReason
models = {
    "RelationVKG": reasoning_vkg.RelationVKG,
    'BottomUp': bottom_up.BottomUp,
    'RN': relationalReason.RNModel
}