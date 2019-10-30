if __name__ == "__main__":
    model.eval()
    accuracy = 0
    for images, labels in testloader:
        log_ps = model(images)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("Test Accuracy:{:.3f}".format(accuracy/len(testloader)))