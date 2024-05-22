from typing import Any, Dict, List, Optional, Type

from pydantic.v1 import BaseModel, Field, validator
from redis import Redis
from redisvl.schema import IndexInfo, IndexSchema, StorageType
from ulid import ULID


def generate_ulid():
    return str(ULID())


class RedisConfig(BaseModel):
    index_name: str = Field(default_factory=lambda: generate_ulid())
    key_prefix: Optional[str] = None
    redis_url: str = "redis://localhost:6379"
    redis_client: Optional[Redis] = Field(default=None)
    connection_args: Optional[Dict[str, Any]] = Field(default={})
    distance_metric: str = "COSINE"
    indexing_algorithm: str = "FLAT"
    vector_datatype: str = "FLOAT32"
    storage_type: str = "hash"
    id_field: str = "id"
    content_field: str = "text"
    embedding_field: str = "embedding"
    default_tag_separator: str = "|"
    metadata_schema: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    index_schema: Optional[IndexSchema] = Field(default=None, alias="schema")
    schema_path: Optional[str] = None
    return_keys: bool = False
    custom_keys: Optional[List[str]] = None
    embedding_dimensions: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("key_prefix", always=True)
    def set_key_prefix(cls, v, values):
        if v is None:
            return values["index_name"]
        return v

    @validator("index_schema", "schema_path", "metadata_schema")
    def check_schema_options(cls, v, values, field):
        options = [
            values.get("index_schema"),
            values.get("schema_path"),
            values.get("metadata_schema"),
        ]
        if sum(1 for option in options if option is not None) > 1:
            raise ValueError(
                "Only one of 'index_schema', 'schema_path', or 'metadata_schema' can be specified"
            )
        return v

    @classmethod
    def from_kwargs(cls: Type["RedisConfig"], **kwargs: Any) -> "RedisConfig":
        """
        Create a RedisConfig object with default values, overwritten by provided kwargs.
        """
        # Get the default values from the class attributes
        default_config = {}
        for field_name, field in cls.__fields__.items():
            if field.default is not None:
                default_config[field_name] = field.default
            elif field.default_factory is not None:
                default_config[field_name] = field.default_factory()

        # Handle special case for 'schema' argument
        if "schema" in kwargs:
            kwargs["index_schema"] = kwargs.pop("schema")

        # Update default_config with any provided kwargs
        default_config.update(kwargs)

        # Create and return the RedisConfig object
        return cls(**default_config)

    @classmethod
    def from_schema(cls, schema: IndexSchema, **kwargs):
        return cls(schema=schema, **kwargs)

    @classmethod
    def from_yaml(cls, schema_path: str, **kwargs):
        return cls(schema_path=schema_path, **kwargs)

    @classmethod
    def with_metadata_schema(cls, metadata_schema: List[Dict[str, Any]], **kwargs):
        return cls(metadata_schema=metadata_schema, **kwargs)

    @classmethod
    def from_existing_index(cls, index_name: str, redis: Redis) -> "RedisConfig":
        index_info = redis.execute_command("FT.INFO", index_name)

        # Parse the index information
        index_def = dict(zip(index_info[::2], index_info[1::2]))

        config = cls(
            index_name=index_name,
            key_prefix=index_def.get("prefix", ""),
            storage_type=index_def.get("index_definition", {}).get("key_type", "hash"),
        )

        # Parse fields
        fields = index_def.get("attributes", [])
        metadata_fields = []
        for i in range(0, len(fields), 7):  # Each field has 7 attributes
            field = dict(zip(fields[i : i + 7 : 2], fields[i + 1 : i + 7 : 2]))
            if field["type"] == "VECTOR":
                config.embedding_field = field["identifier"]
                config.distance_metric = field.get("DISTANCE_METRIC", "COSINE").upper()
                config.indexing_algorithm = field.get("TYPE", "FLAT").upper()
                config.vector_datatype = field.get("DATATYPE", "FLOAT32").upper()
                config.embedding_dimensions = int(field.get("DIM", 0))
            elif field["type"] == "TEXT" and field["identifier"] not in [
                config.id_field
            ]:
                config.content_field = field["identifier"]
            elif field["identifier"] not in [
                config.id_field,
                config.embedding_field,
                config.content_field,
            ]:
                metadata_fields.append(
                    {"name": field["identifier"], "type": field["type"].lower()}
                )

        config.metadata_schema = metadata_fields
        return config

    def to_index_schema(self) -> IndexSchema:
        if self.index_schema:
            return self.index_schema
        elif self.schema_path:
            return IndexSchema.from_yaml(self.schema_path)
        else:
            index_info = {
                "name": self.index_name,
                "prefix": self.key_prefix,
                "storage_type": self.storage_type,
            }

            fields = [
                {"name": self.id_field, "type": "tag"},
                {"name": self.content_field, "type": "text"},
                {
                    "name": self.embedding_field,
                    "type": "vector",
                    "attrs": {
                        "dims": self.embedding_dimensions,
                        "distance_metric": self.distance_metric.lower(),
                        "algorithm": self.indexing_algorithm.lower(),
                        "datatype": self.vector_datatype.lower(),
                    },
                },
            ]

            if self.metadata_schema:
                fields.extend(self.metadata_schema)

            return IndexSchema.from_dict({"index": index_info, "fields": fields})

    def redis(self) -> Redis:
        if self.redis_client is not None:
            return self.redis_client
        elif self.redis_url is not None:
            return Redis.from_url(self.redis_url, **self.connection_args)