build:
	cargo build --release

install:
	cp target/release/align-hmm /usr/bin/align-hmm

clean: 
	rm -rf target/