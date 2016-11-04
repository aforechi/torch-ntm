require('gnuplot')

function argmax(x)
	local index = 1
	local max = x[1]
	for i = 2, x:size(1) do
		if x[i] > max then
			index = i
			max = x[i]
		end
	end
	return index, max
end

function print_read_max(model)
	local read_weights = model:get_read_weights()
	local num_heads = model.read_heads
	local fmt = '%-4d %.4f'
	if num_heads == 1 then
		printf(fmt .. '\n', argmax(read_weights))
	else
		local s = ''
		for i = 1, num_heads do
			s = s .. string.format(fmt, argmax(read_weights[i]))
			if i < num_heads then s = s .. ' | ' end
		end
		print(s)
	end
end

function print_write_max(model)
	local write_weights = model:get_write_weights()
	local num_heads = model.write_heads
	local fmt = '%-4d %.4f'
	if num_heads == 1 then
		printf(fmt .. '\n', argmax(write_weights))
	else
		local s = ''
		for i = 1, num_heads do
			s = s .. string.format(fmt, argmax(write_weights[i]))
			if i < num_heads then s = s .. ' | ' end
		end
		print(s)
	end
end

local function _save_plots_copy(ioExample, output)
  gnuplot.pngfigure('logs/target.png')
  gnuplot.figure(1)
  gnuplot.raw('set cbrange [0:1]')
  gnuplot.imagesc(ioExample.target, 'color')

  gnuplot.pngfigure('logs/output.png')
  gnuplot.figure(2)
  gnuplot.raw('set cbrange [0:1]')
  gnuplot.imagesc(output, 'color')

  gnuplot.pngfigure('logs/error.png')
  gnuplot.figure(3)
  gnuplot.raw('set cbrange[-1:1]')
  gnuplot.imagesc(torch.csub(output, ioExample.target), 'color')

  gnuplot.plotflush(1)
  gnuplot.plotflush(2)
  gnuplot.plotflush(3)
end

local function _save_plots_recall(ioExample, output)
    _save_plots_copy(ioExample, output)
end

function save_plots(taskName, ioExample, output)
  local savePltCmds = {
    copy = _save_plots_copy,
    recall = _save_plots_recall
  }
  savePltCmds[taskName](ioExample, output)
end
