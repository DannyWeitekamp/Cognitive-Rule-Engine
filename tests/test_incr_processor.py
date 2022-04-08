from cre.fact import define_fact
from cre.context import cre_context
from cre.incr_processor import IncrProcessor, IncrProcessorType
from cre.memory import Memory





def test_get_changes():
    with cre_context("test_get_changes"):
        BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})

        mem = Memory()
        ip = IncrProcessor(mem,0)

        b1 = BOOP("A", 1)
        b2 = BOOP("A", 1)
        b3 = BOOP("A", 1)
        mem.declare(b1)
        mem.declare(b2)
        mem.declare(b3)
        mem.modify(b3, "A", "Q")

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

        mem.modify(b1,"A", "Q")
        mem.modify(b1,"A", "V")
        mem.modify(b1,"B", 7)
        mem.modify(b2,"A", "Q")
        mem.modify(b3,"B", 7)
        
        changes = ip.get_changes()
        assert len(changes) == 3
        assert changes[0].was_modified == True
        assert len(changes[0].a_ids) == 2
        assert changes[1].was_modified == True
        assert len(changes[1].a_ids) == 1
        assert changes[2].was_modified == True
        assert len(changes[2].a_ids) == 1

        mem.retract(b1)
        mem.retract(b2)
        mem.retract(b3)

        b1 = BOOP("A", 1)
        b2 = BOOP("A", 1)
        b3 = BOOP("A", 1)

        mem.declare(b3)
        mem.declare(b2)
        mem.declare(b1)

        mem.modify(b1,"A", "Q")
        mem.modify(b1,"A", "V")
        mem.modify(b1,"B", 7)
        mem.modify(b2,"A", "Q")
        mem.modify(b3,"B", 7)

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
