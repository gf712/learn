#include "eigensolver.hpp"

using namespace ngraph;

constexpr NodeTypeInfo op::JacobiSVD::type_info;

op::JacobiSVD::JacobiSVD(const Output<Node>& input0) : Op({input0})
{
	constructor_validate_and_infer_types(); // Runs shape inference on this op
	                                        // by invoking
	                                        // validate_and_infer_types()
}
// Shape inference routine
// First output has the same shape and type as the first input
// Second output has the same shape as second input but the same type as first
// input
void op::JacobiSVD::validate_and_infer_types()
{
	// auto output0_shape = get_input_partial_shape(0);
	auto input0_shape = input(0).get_partial_shape();
	auto output_et = get_input_element_type(0);
	auto output0_shape = PartialShape({input0_shape[0]});
	auto output1_shape = input0_shape;
	set_output_type(0, output_et, output0_shape);
	set_output_type(1, output_et, output1_shape);
}
std::shared_ptr<Node>
op::JacobiSVD::copy_with_new_args(const NodeVector& new_args) const
{
	check_new_args_count(this, new_args);
	return std::make_shared<JacobiSVD>(new_args.at(0));
}
bool op::JacobiSVD::visit_attributes(AttributeVisitor& visitor)
{
	// Visit each of the attributes in the object that would need to be
	// serialized. A serializer implementation of `visitor` might get
	// m_attribute and store it with the string 'attribute' while
	// a deserializer might read the value of a string 'attribute' and
	// set m_attribute accordingly.
	// visitor.on_attribute("attribute", m_attribute);
	return true;
}