import sqlalchemy as sq
from sqlalchemy.orm import DeclarativeBase, relationship,deferred

class Base(DeclarativeBase):
    __abstract__ = True
    id = sq.Column(sq.INTEGER, primary_key=True, autoincrement=True)
    created_time = sq.Column(sq.TIMESTAMP(timezone=True), server_default=sq.func.now())

    def __repr__(self):
        repr_ignore = getattr(self, "__repr_ignore__", [])
        attrs = []
        for key, value in self.__dict__.items():
            if not key.startswith("_") and not callable(value):
                if key not in repr_ignore:
                    attrs.append(f"{key}={value}")

        repr_str = ", ".join(attrs)
        return f"<{self.__class__.__name__}({repr_str})"


class Image(Base):
    __tablename__ = "images"
    file_name = sq.Column(sq.VARCHAR(255))
    bucket_name = sq.Column(sq.VARCHAR(255), index=True)
    width = sq.Column(sq.INTEGER)
    height = sq.Column(sq.INTEGER)

    annotations = relationship("Annotation", back_populates="image")
    __repr_ignore__ = ["annotations"]


class Category(Base):
    __tablename__ = "categories"
    super_category = sq.Column(sq.VARCHAR(255))
    name = sq.Column(sq.VARCHAR(255))

    annotations = relationship("Annotation", back_populates="category")

    __table_args__ = (
        sq.UniqueConstraint("super_category", "name", name="ux_categories"),
    )
    __repr_ignore__ = ["annotations"]


class Annotation(Base):
    __tablename__ = "annotations"
    image_id = sq.Column(sq.INTEGER, sq.ForeignKey("images.id"))
    category_id = sq.Column(sq.INTEGER, sq.ForeignKey("categories.id"))
    iscrowd = sq.Column(sq.BOOLEAN)
    segmentation = deferred(sq.Column(sq.TEXT))  # store segmentation as RLE format
    bbox = sq.Column(sq.ARRAY(sq.FLOAT))  # x,y,w,h
    area = sq.Column(sq.FLOAT)

    category = relationship("Category", back_populates="annotations")
    image = relationship("Image", back_populates="annotations")

    __repr_ignore__ = ["segmentation", "image", "category"]


class DataSet(Base):
    __tablename__ = "datasets"
    dataset_name = sq.Column(sq.VARCHAR(255))
    dataset_type = sq.Column(sq.VARCHAR(255))
    image_ids = sq.Column(sq.ARRAY(sq.INTEGER))

    __table_args__ = (
        sq.UniqueConstraint("dataset_name", "dataset_type", name="ux_datasets"),
    )
