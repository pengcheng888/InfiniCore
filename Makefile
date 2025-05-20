name      := pro
workdir   := workspace111
python      := /home/ubuntu/miniconda3/envs/py310torch260/bin/python

# /home/ubuntu/miniconda3/envs/py310torch260/bin/python
# @mkdir -p build && cd build && cmake .. && make -j16
build:

	@echo "------------------- xmake f -v  --------------------"
	@xmake f -v

	@echo "------------------- xmake f --cpu=true -cv  --------------------"
	@xmake f --cpu=true -cv

	@echo "------------------- xmake f --nv-gpu=true --cuda=${CUDA_HOME} -cv  --------------------"
	@xmake f --nv-gpu=true --cuda=${CUDA_HOME} -cv

	@echo "------------------- xmake build && xmake install  --------------------"
	@xmake build && xmake install

build-test:
	@echo "------------------- xmake build infiniop-test  --------------------"
	@xmake build infiniop-test

	@echo "------------------- testcases.add   --------------------"
	@cd ./test/infiniop-test/ && $(python) -m test_generate.testcases.add 

	@echo "------------------- testcases.swiglu   --------------------"
	@cd ./test/infiniop-test/ && $(python) -m test_generate.testcases.swiglu 



run-test:
	# @echo "------------------- infiniop-test swiglu.gguf --nvidia --warmup 2 --run 10   --------------------"
	# @./build/linux/x86_64/release/infiniop-test ./test/infiniop-test/swiglu.gguf --nvidia --warmup 2 --run 10
	@echo "------------------- infiniop-test add.gguf --nvidia --warmup 2 --run 10   --------------------"
	@./build/linux/x86_64/release/infiniop-test ./test/infiniop-test/add.gguf --nvidia --warmup 2 --run 10


# @make build && cd $(workdir) && ./$(name)
run:
	# @echo "\n\n\n ------------------- python add_v2.py --nvidia --profile -------------------- \n"
	# @$(python)  test/infiniop/add_v2.py  --nvidia --profile

	# @echo "\n\n\n ------------------- python swiglu_v2.py --nvidia --profile -------------------- \n"
	# @$(python) test/infiniop/swiglu_v2.py  --nvidia --profile
	
	@echo "\n\n\n ------------------- python attention_v2.py --nvidia --profile -------------------- \n"
	@$(python)  test/infiniop/attention_v2.py  --nvidia --profile



# 定义清理指令
clean:
	@xmake clean
	@rm -rf  ./build $(workdir)/$(name)

# 防止符号被当做文件
.PHONY : build run clean build-test run-test