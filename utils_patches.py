def extract_image_patches(images, ksizes, strides, rates, padding='same'):
        assert len(images.size()) == 4
        assert padding in ['same', 'valid']
        batch_size, channel, height, width = images.size()
        if padding == 'same':
            images = same_padding(images, ksizes, strides, rates)
        elif padding == 'valid':
            pass
        else:
            raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))
        unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
        patches = unfold(images)
        return patches
