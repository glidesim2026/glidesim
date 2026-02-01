from glidesim.strategies.fixed import FixedWithdrawalStrategy


def test_returns_constant_amount():
    strategy = FixedWithdrawalStrategy(withdrawal_amount=40_000)
    assert strategy.get_withdrawal(1_000_000, 1.0, 0) == 40_000
    assert strategy.get_withdrawal(500_000, 2.0, 0) == 40_000


def test_ignores_inflation():
    strategy = FixedWithdrawalStrategy(withdrawal_amount=40_000)
    assert strategy.get_withdrawal(1_000_000, 1.5, 0) == 40_000
    assert strategy.get_withdrawal(1_000_000, 2.0, 0) == 40_000


def test_ignores_portfolio_value():
    strategy = FixedWithdrawalStrategy(withdrawal_amount=40_000)
    assert strategy.get_withdrawal(2_000_000, 1.0, 0) == 40_000
    assert strategy.get_withdrawal(100_000, 1.0, 0) == 40_000


def test_caps_withdrawal_at_portfolio_value():
    strategy = FixedWithdrawalStrategy(withdrawal_amount=40_000)
    assert strategy.get_withdrawal(20_000, 1.0, 0) == 20_000


def test_returns_zero_when_portfolio_empty():
    strategy = FixedWithdrawalStrategy(withdrawal_amount=40_000)
    assert strategy.get_withdrawal(0, 1.0, 0) == 0
