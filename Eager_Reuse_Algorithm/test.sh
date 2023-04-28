loop=7
i=0
rm -rf result/result*
while [ $loop -gt $i ]
do
	echo =====Graph No.$[$i+1] Memory Reuse under three algorithms=====
	./memory_reuse_test $i
	grep -ri  memory_size result/result$i
	i=$[$i+1]
	echo
done

