import pytest
from bytecheck import canonical_bytes


def test_canonical_object_order():
    assert canonical_bytes({'a': 1, 'b': 2}) == canonical_bytes({'b': 2, 'a': 1})


@pytest.mark.parametrize('a,b', [(1,1.0),(False,0),(-0.0,0.0),([1,2],[2,1]),({'a':1},{'a':2})])
def test_numeric_and_sequence_changes_are_visible(a,b):
    assert canonical_bytes(a) != canonical_bytes(b)


def test_inventory_order_is_part_of_snapshot():
    def snapshot(inv):
        return {'observations':[{'private':{'inventories':[inv]}}]}
    assert canonical_bytes(snapshot({'WHEAT':1,'FERTILIZER':2})) != canonical_bytes(snapshot({'FERTILIZER':2,'WHEAT':1}))


def test_tuple_cannot_masquerade_as_list():
    with pytest.raises(AssertionError):
        canonical_bytes((1,2))
