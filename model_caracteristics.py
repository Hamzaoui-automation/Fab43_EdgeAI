import onnxruntime as ort

session = ort.InferenceSession("model.onnx")

print("=== INPUTS ===")
for inp in session.get_inputs():
    print("Name:", inp.name)
    print("Shape:", inp.shape)
    print("Type:", inp.type)

print("\n=== OUTPUTS ===")
for out in session.get_outputs():
    print("Name:", out.name)
    print("Shape:", out.shape)
    print("Type:", out.type)