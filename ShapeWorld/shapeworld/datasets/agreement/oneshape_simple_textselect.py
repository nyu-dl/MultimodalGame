from shapeworld.dataset import CaptionAgreementDataset, TextSelectionDataset
from shapeworld.generators import RandomAttributesGenerator
from shapeworld.captioners import RegularTypeCaptioner


class OneshapeSimpleTextselect(TextSelectionDataset):

    def __init__(
        self,
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=6,
        vocabulary=('.', 'a', 'an', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'yellow'),
        language=None,
        number_texts=10, 
    ):

        world_generator = RandomAttributesGenerator(
            entity_counts=[1],
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            max_provoke_collision_rate=0.0,
            collision_tolerance=0.0,
            boundary_tolerance=0.0
        )
        world_generator.world_size = 128

        world_captioner = RegularTypeCaptioner(
            existing_attribute_rate=0.0
        )

        super(OneshapeSimpleTextselect, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language, 
            number_texts=number_texts
        )


dataset = OneshapeSimpleTextselect
