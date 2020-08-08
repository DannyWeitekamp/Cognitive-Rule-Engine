import unittest
from numbert.numbalizer import Numbalizer,decode_vectorized
from numba.typed import List, Dict
import numpy as np



class TestDataTransfer(unittest.TestCase):
	def setUp(self):
		self.object_specifications = {
			"InterfaceElement" : {
				"id" : "String",
				"value" : "String",
				"above" : "String",#["String","Reference"],
				"below" : "String",#["String","Reference"],
				"to_left" : "String",#["String","Reference"],
				"to_right" : "String",# ["String","Reference"],
				"x" : "Number",
				"y" : "Number"
			},
			"Trajectory" : {
				"x" : "Number",
				"y" : "Number",
				"z" : "Number",
				"dx" : "Number",
				"dy" : "Number",
				"dz" : "Number",
				"a_x" : "Number",
				"a_y" : "Number",
				"a_z" : "Number",
				"a_dx" : "Number",
				"a_dy" : "Number",
				"a_dz" : "Number",
			}
		}
		self._state = {
			"i0" : {
				"type" : "InterfaceElement",
				"id" : "i0",
				"value" : "9",
				"above" : "",
				"below" : "i1",
				"to_left" : "",
				"to_right" : "",
				"x" : 100,
				"y" : 100,
			},
			"i1" : {
				"type" : "InterfaceElement",
				"id" : "i1",
				"value" : "7",
				"above" : "i0",
				"below" : "",
				"to_left" : "",
				"to_right" : "",
				"x" : 100,
				"y" : 200,
			}
		}

		self._state2 = {
			"a" : {
				"type" : "Trajectory",
				"x" : 1,
				"y" : 2,
				"z" : 3,
				"dx" : 5.5,
				"dy" : 5.9,
				"dz" : 0.4,
				"a_x" : 1,
				"a_y" : 2,
				"a_z" : 3,
				"a_dx" : 5.5,
				"a_dy" : 5.9,
				"a_dz" : 0.4,
			}
		}
		self.numbalizer = Numbalizer()
		self.numbalizer.register_specifications(self.object_specifications)
		self.nb_objs = self.numbalizer.state_to_nb_objects(self._state)
		self.nb_objs2 = self.numbalizer.state_to_nb_objects(self._state2)
		self.nb_objs_mixed = self.numbalizer.state_to_nb_objects({**self._state,**self._state2})

		self.enumerized1 = self.numbalizer.nb_objects_to_enumerized(self.nb_objs)
		self.enumerized2 = self.numbalizer.nb_objects_to_enumerized(self.nb_objs2)
		self.enumerized_mixed = self.numbalizer.nb_objects_to_enumerized(self.nb_objs_mixed)

	def test_register_spec(self):
		numbalizer = Numbalizer()
		numbalizer.register_specifications(self.object_specifications)
		self.assertTrue(hasattr(numbalizer,'jitstructs'))
		self.assertTrue(hasattr(numbalizer,'spec_flags'))
		self.assertTrue("nominal" in numbalizer.spec_flags)
		self.assertTrue((numbalizer.spec_flags['nominal']['InterfaceElement'] == np.array([1,1,1,1,1,1,0,0])).all())
		self.assertTrue((numbalizer.spec_flags['nominal']['Trajectory'] == np.array([0,0,0,0,0,0,0,0,0,0,0,0])).all())

		self.assertTrue('InterfaceElement' in numbalizer.jitstructs)

		self.assertTrue(hasattr(numbalizer.jitstructs['InterfaceElement'],'get_enumerized'))
		self.assertTrue(hasattr(numbalizer.jitstructs['InterfaceElement'],'pack_from_numpy'))
		self.assertTrue(hasattr(numbalizer.jitstructs['InterfaceElement'],'enumerize_nb_objs'))

	def test_state_to_nb_objects(self):
		numbalizer = self.numbalizer
		nb_objs = numbalizer.state_to_nb_objects(self._state)
		nb_objs2 = numbalizer.state_to_nb_objects(self._state2)
		nb_objs_mixed = numbalizer.state_to_nb_objects({**self._state,**self._state2})

		self.assertIsInstance(nb_objs,dict)

		self.assertTrue("InterfaceElement" in nb_objs)
		self.assertTrue("InterfaceElement" in nb_objs_mixed)
		self.assertTrue("Trajectory" in nb_objs2)
		self.assertTrue("Trajectory" in nb_objs_mixed)

		i1 = nb_objs['InterfaceElement']['i1']
		self.assertTupleEqual(('i1','7','i0','','','',100,200),
		 (i1.id, i1.value, i1.above, i1.below, i1.to_left, i1.to_right, i1.x, i1.y))

		i1 = nb_objs_mixed['InterfaceElement']['i1']
		self.assertTupleEqual(('i1','7','i0','','','',100,200),
		 (i1.id, i1.value, i1.above, i1.below, i1.to_left, i1.to_right, i1.x, i1.y))

		a = nb_objs2['Trajectory']['a']
		self.assertTupleEqual((1., 2., 3., 5.5, 5.9, 0.4, 1., 2., 3., 5.5, 5.9, 0.4),
		 (a.x, a.y, a.z, a.dx, a.dy, a.dz, a.a_x,a.a_y,a.a_z,a.a_dx,a.a_dy,a.a_dz))

		a = nb_objs_mixed['Trajectory']['a']
		self.assertTupleEqual((1., 2., 3., 5.5, 5.9, 0.4, 1., 2., 3., 5.5, 5.9, 0.4),
		 (a.x, a.y, a.z, a.dx, a.dy, a.dz, a.a_x,a.a_y,a.a_z,a.a_dx,a.a_dy,a.a_dz))

	def test_nb_objects_to_enumerized(self):
		numbalizer = self.numbalizer
		enumerized1 = numbalizer.nb_objects_to_enumerized(self.nb_objs)
		enumerized2 = numbalizer.nb_objects_to_enumerized(self.nb_objs2)
		enumerized_mixed = numbalizer.nb_objects_to_enumerized(self.nb_objs_mixed)
		
		# self.assertIsInstance(enumerized1,)
		self.assertTrue("InterfaceElement" in enumerized1)
		self.assertTrue("InterfaceElement" in enumerized_mixed)
		self.assertTrue("Trajectory" in enumerized2)
		self.assertTrue("Trajectory" in enumerized_mixed)

		i0 = enumerized_mixed['InterfaceElement']['i0']
		i1 = enumerized_mixed['InterfaceElement']['i1']
		self.assertEqual(len(i0),8)
		self.assertEqual(len(i1),8)
		self.assertNotEqual(i0[0],i1[0])
		self.assertNotEqual(i0[1],i1[1])
		self.assertNotEqual(i0[2],i1[2])
		self.assertNotEqual(i0[3],i1[3])
		self.assertEqual(i0[4],i1[4])
		self.assertEqual(i0[5],i1[5])
		self.assertEqual(i0[6],i1[6])
		self.assertNotEqual(i0[7],i1[7])

		a = enumerized_mixed['Trajectory']['a']
		self.assertEqual(len(a),12)
		self.assertEqual(a[0],a[6])
		self.assertEqual(a[1],a[7])
		self.assertEqual(a[2],a[8])
		self.assertNotEqual(a[0],a[3])
		self.assertNotEqual(a[1],a[4])
		self.assertNotEqual(a[2],a[5])

	def test_enumerized_to_vectorized(self):
		numbalizer = self.numbalizer

		inp1 = List()
		inp1.append(self.enumerized1)

		inp_m = List()
		inp_m.append(self.enumerized_mixed)

		stateA = {"ie" + str(i) : self._state['i0'] for i in range(10)}
		stateB = {"ie" + str(i+3) : self._state['i1'] for i in range(10)}
		enumerized_A = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(stateA))
		enumerized_B = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(stateB))

		inp_offset = List()
		inp_offset.append(enumerized_A)
		inp_offset.append(enumerized_B)
		
		vectorized1 = numbalizer.enumerized_to_vectorized(inp1)
		vectorized_mixed = numbalizer.enumerized_to_vectorized(inp_m)
		vectorized_offset = numbalizer.enumerized_to_vectorized(inp_offset)


		ie_nominal_width = len(vectorized1['nominal'][0])//2
		ie_continous_width = len(vectorized1['continuous'][0])//2
		# print(len(vectorized1['nominal'][0]))
		self.assertEqual(np.sum(vectorized1['nominal'][0]),12)
		self.assertEqual(len(vectorized1['nominal'][0]),20)
		self.assertEqual(len(vectorized1['continuous'][0]),4)
		self.assertTrue((vectorized1['continuous'][0] == np.array([100,100,100,200])).all())

		self.assertEqual(len(vectorized_mixed['nominal'][0]),ie_nominal_width*2)
		self.assertEqual(len(vectorized_mixed['continuous'][0]),4+12)
		
		# print(ie_continous_width ,len(vectorized_offset['nominal'][0][-((ie_nominal_width*3)):]), vectorized_offset['nominal'][0][-((ie_nominal_width*3)):])
		# print(vectorized_offset['continuous'][0][-(ie_continous_width*3):])
		self.assertTrue((vectorized_offset['nominal'][0][-(ie_nominal_width*3):] == 255).all())
		self.assertTrue((vectorized_offset['nominal'][1][:ie_nominal_width*3] == 255).all())
		self.assertTrue(np.isnan(vectorized_offset['continuous'][0][-(ie_continous_width*3):]).all())
		self.assertTrue(np.isnan(vectorized_offset['continuous'][1][:ie_continous_width*3]).all())

		# print("WEE")
		# print(vectorized1['nominal'])
		# print(vectorized1['continuous'])
		
class TestDataMapBack(unittest.TestCase):
	def setUp(self):
		TestDataTransfer.setUp(self)
		numbalizer = self.numbalizer
		self.stateA = {"ie" + str(i) : self._state['i0'] for i in range(10)}
		self.stateB = {"ie" + str(i+3) : self._state['i1'] for i in range(10)}
		self.enumerized_A = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(self.stateA))
		self.enumerized_B = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(self.stateB))
		inp_offset = List()
		inp_offset.append(self.enumerized_A)
		inp_offset.append(self.enumerized_B)
		self.vectorized_offset = numbalizer.enumerized_to_vectorized(inp_offset)

	def test_decode_vectorized(self):
		numbalizer = self.numbalizer
		inp_offset = List()
		inp_offset.append(self.enumerized_A)
		inp_offset.append(self.enumerized_B)
		vectorized_offset = numbalizer.enumerized_to_vectorized(inp_offset,return_inversion_data=True)
		print(vectorized_offset['inversion_data'])

		L = (len(self.stateA)+3)*10
		print("LEN",L)
		out = decode_vectorized(np.arange(L),np.ones(L),vectorized_offset['inversion_data'])

		correct_vals = [(0,'i0'),(0,'i1'),(1,'9'),(1,'7'),(2,''),(2,'i0'),(3,'i1'),(3,''),(4,''),(5,'')]
		for i,(typ,name,attr,attr_v) in enumerate(out):
			cv = correct_vals[i%10]
			attr_v = numbalizer.string_backmap[attr_v]
			self.assertTupleEqual((typ,name,attr,attr_v),("InterfaceElement",'ie%s'%(str(i//10)), cv[0],cv[1]))



