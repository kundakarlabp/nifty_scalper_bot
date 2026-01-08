from enum import Enum, auto


class FSMState(Enum):
    IDLE = auto()
    SIGNAL_QUALIFIED = auto()
    RISK_APPROVED = auto()
    ORDER_PENDING = auto()
    POSITION_OPEN = auto()
    EXIT_PENDING = auto()


LEGAL_TRANSITIONS = {
    FSMState.IDLE: {FSMState.SIGNAL_QUALIFIED},
    FSMState.SIGNAL_QUALIFIED: {FSMState.RISK_APPROVED, FSMState.IDLE},
    FSMState.RISK_APPROVED: {FSMState.ORDER_PENDING, FSMState.IDLE},
    FSMState.ORDER_PENDING: {FSMState.POSITION_OPEN, FSMState.IDLE},
    FSMState.POSITION_OPEN: {FSMState.EXIT_PENDING},
    FSMState.EXIT_PENDING: {FSMState.IDLE},
}


def is_transition_allowed(current: FSMState, nxt: FSMState) -> bool:
    return nxt in LEGAL_TRANSITIONS.get(current, set())
