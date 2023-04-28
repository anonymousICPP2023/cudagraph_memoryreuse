memory_reuse_test:common.o short_lifetime_first.o large_tensor_first.o eager_reuse.o test.o
	gcc -g -o memory_reuse_test common.o short_lifetime_first.o large_tensor_first.o eager_reuse.o test.o

test.o: test.c
	gcc -c test.c

common.o: common.c
	gcc -c common.c


eager_reuse.o:eager_reuse.c
	gcc -c eager_reuse.c   


short_lifetime_first.o:short_lifetime_first.c
	gcc -c short_lifetime_first.c   


large_tensor_first.o:large_tensor_first.c 
	gcc -c large_tensor_first.c

clean:
	rm -rf *.o
