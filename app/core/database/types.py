from sqlalchemy import BigInteger, Integer


PrimaryKeyInt = BigInteger().with_variant(Integer, "sqlite")

