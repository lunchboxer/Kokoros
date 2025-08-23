# syntax=docker/dockerfile:1
FROM rust:1.86.0-slim-bookworm AS builderrs

RUN apt-get update -qq && apt-get install -qq -y wget pkg-config libssl-dev clang git cmake && rustup component add rustfmt

WORKDIR /app

COPY . .
COPY Cargo.toml .
COPY Cargo.lock .

# Use make to download data and build
RUN make data && make build

FROM debian:sid-slim AS runner

WORKDIR /app

COPY --from=builderrs /app/target/release/koko ./target/release/koko
COPY --from=builderrs /app/data ./data
COPY --from=builderrs /app/checkpoints ./checkpoints

RUN chmod +x ./target/release/koko && apt-get update -qq && apt-get install -qq -y pkg-config libssl-dev 

EXPOSE 3000

ENTRYPOINT [ "./target/release/koko" ]
