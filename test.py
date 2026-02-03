from loss import loss
import torch

# Assuming your compute_loss and giou functions are defined above...


def test_loss():
    # Setup dimensions
    batch_size = 2
    S = 7
    B = 2
    C = 20

    # 1. Create Dummy Target (N, S, S, 25)
    # [0:20] classes, [20] conf, [21:25] x, y, w, h
    target = torch.zeros((batch_size, S, S, 25))

    # Place an object in cell (3,3) for the first image
    target[0, 3, 3, 20] = 1.0  # Object exists
    target[0, 3, 3, 10] = 1.0  # Class index 10
    target[0, 3, 3, 21:25] = torch.tensor(
        [0.5, 0.5, 0.2, 0.2])  # Centered in cell

    # 2. Create Dummy Prediction (N, S, S, 30)
    # Box1 at 20-24, Box2 at 25-29
    prediction = torch.randn((batch_size, S, S, 30), requires_grad=True)

    # 3. Run the loss
    try:
        losses = loss(target, prediction)
        bbox, obj, noobj, cls, total = losses

        print("✅ Loss calculation successful!")
        print(f"Total Loss: {total.item():.4f}")
        print(f"BBox Loss:  {bbox.item():.4f}")
        print(f"Obj Loss:   {obj.item():.4f}")
        print(f"Cls Loss:   {cls.item():.4f}")

        # 4. Check Gradient Flow
        total.backward()
        print("✅ Backpropagation successful! Gradients computed.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")


if __name__ == "__main__":
    test_loss()
