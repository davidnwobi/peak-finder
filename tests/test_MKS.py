from MarketStructure import MKS
import datetime as dt
import copy
import pytest
import random

node1 = MKS.Peak(dt.datetime.now(), 10.0)
node2 = MKS.Trough(dt.datetime.now(), 20.0)
node3 = MKS.Peak(dt.datetime.now(), 30.0)


def test_Direction():
    assert MKS.Direction.FORWARD == "forward"
    assert MKS.Direction.BACKWARD == "backward"


def test_peak_eq():
    peak1 = MKS.Peak(dt.datetime.now(), 10.0)
    peak2 = copy.copy(peak1)

    assert peak1 == peak2


def test_trough_eq():
    trough1 = MKS.Trough(dt.datetime.now(), 10.0)
    trough2 = copy.copy(trough1)

    assert trough1 == trough2


def test_append_node_to_market_structure():
    mkt_structure = MKS.MarketStructure()
    mkt_structure.append(node1)
    mkt_structure.append(node2)
    assert node1.next == node2
    assert node2.prev == node1


def test_pop_from_market_structure():
    mkt_structure = MKS.MarketStructure()
    mkt_structure.append(node1)
    mkt_structure.append(node2)
    assert mkt_structure.popback() == node2
    assert mkt_structure.popback() == node1
    with pytest.raises(IndexError):
        mkt_structure.popback()


def test_insert_to_market_structure():
    mkt_structure = MKS.MarketStructure()
    mkt_structure.append(node1)
    mkt_structure.append(node2)
    mkt_structure.insert(1, node3)
    assert mkt_structure.popback() == node2
    assert mkt_structure.popback() == node3
    assert mkt_structure.popback() == node1
    with pytest.raises(IndexError):
        mkt_structure.popback()


def test_find_in_market_structure():
    mkt_structure = MKS.MarketStructure()
    mkt_structure.append(node1)
    mkt_structure.append(node2)
    mkt_structure.append(node3)
    assert mkt_structure.find(lambda x: x == node1) == node1
    assert mkt_structure.find(lambda x: x == node2) == node2
    assert mkt_structure.find(lambda x: x == node3) == node3
    assert mkt_structure.find(lambda x: x == 10.0) is None


def test_index_of_node_in_market_structure():
    mkt_structure = MKS.MarketStructure()
    mkt_structure.append(node1)
    mkt_structure.append(node2)
    mkt_structure.append(node3)
    assert mkt_structure.index(node1) == 0
    assert mkt_structure.index(node2) == 1
    assert mkt_structure.index(node3) == 2
    with pytest.raises(ValueError):
        mkt_structure.index(MKS.Peak())


def test_delete_from_market_structure():
    mkt_structure = MKS.MarketStructure()
    mkt_structure.append(node1)
    mkt_structure.append(node2)
    mkt_structure.append(node3)
    mkt_structure.remove(direction="forward", start_node=node2, end_node=node2)
    assert mkt_structure.popback() == node3
    assert mkt_structure.popback() == node1
    with pytest.raises(IndexError):
        mkt_structure.popback()


def test_remove_from_market_structure():
    mkt_structure = MKS.MarketStructure()
    mkt_structure.append(node1)
    mkt_structure.append(node2)
    mkt_structure.append(node3)
    mkt_structure.remove(direction="backward", start_node=node2, end_node=node2)
    assert mkt_structure.popback() == node3
    assert mkt_structure.popback() == node1
    with pytest.raises(IndexError):
        mkt_structure.popback()




def test_remove_from_market_structure_slice():
    mkt_structure = MKS.MarketStructure()
    mkt_structure.append(node1)
    mkt_structure.append(node2)
    mkt_structure.append(node3)
    mkt_structure.remove(direction="forward", start_node=node2, end_node=node3)
    assert mkt_structure.popback() == node1
    with pytest.raises(IndexError):
        mkt_structure.popback()


@pytest.mark.parametrize("node_list",
                         [([10.0, 20.0, 30.0]),
                          ([(dt.datetime.now(), 10.0), (dt.datetime.now() + dt.timedelta(seconds=1), 20.0),
                            (dt.datetime.now() + dt.timedelta(seconds=2), 30.0)]),

                          ([(10.0, True), (20.0, True), (30.0, True)]),
                          ([(dt.datetime.now(), 10.0, True), (dt.datetime.now() + dt.timedelta(seconds=1), 20.0, True),
                            (dt.datetime.now() + dt.timedelta(seconds=2), 30.0, True)])])
def test_fill_market_structure(node_list):
    mkt_structure = MKS.MarketStructure()
    mkt_structure.fill_from_list(node_list)
    assert mkt_structure.popback().value == 30.0
    assert mkt_structure.popback().value == 20.0
    assert mkt_structure.popback().value == 10.0
    with pytest.raises(IndexError):
        mkt_structure.popback()


@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)])])
def test_find_in_market_structure2(node_list):
    # Find first node with value > 50.0:
    mkt_structure = MKS.MarketStructure()
    mkt_structure.fill_from_list(node_list)
    assert mkt_structure.find(lambda x: x.value > 50.0).value == 60.0
    # Find first node with value > 50.0 and is_peak == True:
    assert mkt_structure.find(lambda x: x.value > 50.0 and isinstance(x, MKS.Peak)).value == 60.0
    # Find first node with value > 50.0 and is_peak == False:
    assert mkt_structure.find(lambda x: x.value > 50.0 and not isinstance(x, MKS.Peak)).value == 70.0
    # Find first node with value > 50.0 and is_permanent == True:
    assert mkt_structure.find(lambda x: x.value > 50.0 and x.is_permanent).value == 70.0
    # Find last non permanent node:
    assert mkt_structure.find(lambda x: not x.is_permanent, direction="backward").value == 100.0


@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)]),
                                       ([(random.randint(0, 1000), bool(random.randint(0, 1))) for i in range(10)])])
def test_min_element_in_market_structure(node_list):
    mkts = MKS.MarketStructure()
    mkts.fill_from_list(node_list)
    min_node = mkts.min_element()
    assert min_node.value == min(node_list, key=lambda x: x[0])[0]


@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)]),
                                       ([(random.randint(0, 1000), bool(random.randint(0, 1))) for i in range(10)])])
def test_min_element_in_market_structure_reverse(node_list):
    mkts = MKS.MarketStructure()
    mkts.fill_from_list(node_list)
    min_node = mkts.min_element(direction="backward")
    assert min_node.value == min(node_list, key=lambda x: x[0])[0]



@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)]),
                                       ([(random.randint(0, 1000), bool(random.randint(0, 1))) for i in range(10)])])
def test_max_element_in_market_structure(node_list):
    mkts = MKS.MarketStructure()
    mkts.fill_from_list(node_list)
    max_node = mkts.max_element()
    assert max_node.value == max(node_list, key=lambda x: x[0])[0]

@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)]),
                                       ([(random.randint(0, 1000), bool(random.randint(0, 1))) for i in range(10)])])
def test_max_element_in_market_structure_reverse(node_list):
    mkts = MKS.MarketStructure()
    mkts.fill_from_list(node_list)
    max_node = mkts.max_element(direction="backward")
    assert max_node.value == max(node_list, key=lambda x: x[0])[0]



@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)]),
                                       ([(random.randint(0, 1000), bool(random.randint(0, 1))) for i in range(10)])])
def test_transform_market_structure(node_list):
    mkts = MKS.MarketStructure()
    mkts.fill_from_list(node_list)

    def transform_func(x):
        x.value += 1.0

    mkts.transform(transform_func)
    assert all([x.value == y[0] + 1.0 for x, y in zip(mkts, node_list)])



@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)])])
def test_transform_market_structure_reverse(node_list):
    mkts = MKS.MarketStructure()
    mkts.fill_from_list(node_list)

    def transform_func(x):
        x.value += 1.0

    node50 = mkts.find(lambda x: x.value == 50.0)
    node80 = mkts.find(lambda x: x.value == 80.0, direction="forward", start_node=node50)
    mkts.transform(transform_func, direction="backward", start_node=node80, end_node=node50)
    sol = [x.value == y[0] + 1.0 for x, y in zip(mkts, node_list) if 50.0 <= y[0] <= 80.0]
    assert all(sol)


@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)])])
def test_slice_list_market_structure_reverse(node_list):
    mkts = MKS.MarketStructure()
    mkts.fill_from_list(node_list)
    node50 = mkts.find(lambda x: x.value == 50.0)
    node80 = mkts.find(lambda x: x.value == 80.0, direction="forward", start_node=node50)
    mks2 = mkts.slice_list(start_node=node50, end_node=node80)
    assert len(mkts) == len(node_list)
    mks3 = mkts.slice_list(start_node=node50, end_node=node80, inplace=True)
    l1 = len(mkts)
    l2 = len(node_list)
    assert l1 == l2 - 4

    assert all([x == y for x, y in zip(mks2, mks3)])




@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)])])
def test_extend_market_structure(node_list):
    mkts = MKS.MarketStructure()
    mkts.fill_from_list(node_list)
    mkts2 = MKS.MarketStructure()
    mkts2.fill_from_list(node_list)
    mkts.extend(mkts2)
    assert all([x.value == y[0] for x, y in zip(mkts, node_list + node_list)])

@pytest.mark.parametrize("node_list", [([(i * 10.0, bool(i % 2)) for i in range(11)])])
def test_erase_market_structure(node_list):
    mkts = MKS.MarketStructure()
    mkts.fill_from_list(node_list)
    mkts.erase(mkts.head, mkts.tail)
    assert len(mkts) == 0
    mkts.fill_from_list(node_list)
    node50 = mkts.find(lambda x: x.value == 50.0)
    mkts.erase(node50, mkts.tail)
    assert len(mkts) == 5
