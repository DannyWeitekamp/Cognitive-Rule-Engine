from cre.fact import define_fact
from cre.context import cre_context
from cre.incr_processor import IncrProcessor, IncrProcessorType
from cre.memset import MemSet





def test_get_changes():
    with cre_context("test_get_changes"):
        BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})

        ms = MemSet()
        ip = IncrProcessor(ms,0)

        b1 = BOOP("A", 1)
        b2 = BOOP("A", 1)
        b3 = BOOP("A", 1)
        ms.declare(b1)
        ms.declare(b2)
        ms.declare(b3)
        ms.modify(b3, "A", "Q")

        changes = ip.get_changes()
        assert len(changes) == 3
        assert changes[0].was_declared == True
        assert changes[1].was_declared == True
        assert changes[2].was_declared == True

        # Declarations clobber modifications
        assert changes[2].was_modified == False
        
        changes = ip.get_changes()
        # Get changes exhausts changes by default
        assert len(changes) == 0

        ms.modify(b1,"A", "Q")
        ms.modify(b1,"A", "V")
        ms.modify(b1,"B", 7)
        ms.modify(b2,"A", "Q")
        ms.modify(b3,"B", 7)
        
        changes = ip.get_changes()
        assert len(changes) == 3
        assert changes[0].was_modified == True
        assert len(changes[0].a_ids) == 2
        assert changes[1].was_modified == True
        assert len(changes[1].a_ids) == 1
        assert changes[2].was_modified == True
        assert len(changes[2].a_ids) == 1

        ms.retract(b1)
        ms.retract(b2)
        ms.retract(b3)

        b1 = BOOP("A", 1)
        b2 = BOOP("A", 1)
        b3 = BOOP("A", 1)

        ms.declare(b3)
        ms.declare(b2)
        ms.declare(b1)

        ms.modify(b1,"A", "Q")
        ms.modify(b1,"A", "V")
        ms.modify(b1,"B", 7)
        ms.modify(b2,"A", "Q")
        ms.modify(b3,"B", 7)

        changes = ip.get_changes()
        assert len(changes) == 3
        # If retraction and declaration both occur they are both true
        assert changes[0].was_declared == True
        assert changes[0].was_retracted == True
        assert changes[0].was_modified == False
        assert changes[1].was_declared == True
        assert changes[1].was_retracted == True
        assert changes[1].was_modified == False
        assert changes[2].was_declared == True
        assert changes[2].was_retracted == True
        assert changes[2].was_modified == False



if __name__ == "__main__":
    test_get_changes()
