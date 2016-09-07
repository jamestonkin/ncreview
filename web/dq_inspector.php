<?php
try{
	$dir = $_REQUEST['dir'];
	$var = $_REQUEST['var'];
	$beg = $_REQUEST['beg'];
	$end = $_REQUEST['end'];
	$version = $_REQUEST['version'];

	// Ensure each parameter conforms to an expected regex

	if (preg_match('/^[a-zA-Z0-9\-\.\/]+\z/', $dir) === false) {
		throw new Exception('Directory argument does not match expected pattern');
	}
	else if (preg_match('/^\w+\z/', $var) === false) {
		throw new Exception('variable name argument does not match expected pattern');	
	}
	else if (preg_match('/^\d{8}(.\d{6})\z/', $beg) === false) {
		throw new Exception('Begin date argument does not match expected pattern');
	}
	else if (preg_match('/^\d{8}(.\d{6})\z/', $end) === false) {
		throw new Exception('End date argument does not match expected pattern');
	}
	if (isset($version) && $version !== 'old' && $version !== 'new') {
		throw new Exception('version argument is invalid');
	}
	// get datastrem name
	
	$temp = explode('/', rtrim($dir, '/'));
	$datastream = array_pop($temp);// what if the datastream isn't in a properly named directory?
	$site = array_pop($temp);
	$path = implode('/', $temp);
	preg_match('/^\d{8}/', $beg, $matches);
	$beg_day = $matches[0];
	preg_match('/^\d{8}/', $end, $matches);
	$end_day = $matches[0];

	// create the write directory
	$write_dir = '/var/tmp/';
	if (isset($version)) {
		$write_dir .= $version;
		if (!file_exists($write_dir)) {
			mkdir($write_dir, 0777, true);
		}
	}

	$cmd = "/apps/tool/bin/dq_inspector_dmf_py3 -d $datastream -r $path -s $beg_day -e $end_day -x $beg:$end -v $var -w $write_dir";
	
	//print $cmd."</br>";
	$saved_home = getenv("HOME");
	putenv("HOME=/data/tmp");
	$saved_user = getenv("USER");
	putenv("USER=ncrevew");
	exec($cmd." 2>&1", $output, $return_val);
	putenv("HOME=$saved_user");
	putenv("USER=$saved_home");
	$output = implode("</br>", $output);
	if ($return_val || strpos($output, 'Errno') !== false) {
		throw new Exception("$cmd</br>failed, with output:</br></br>".$output);
	}

	/* write out the image */
	if (preg_match('/\S*\.png/', $output, $matches) === false) {
		throw new Exception("plot location not found in output: </br>".$output);
	}
	$plot = $matches[0];
	$fp = fopen($plot, 'rb');

	if (!$fp) {
		throw new Exception("$cmd</br>Could not open plot at ".$plot."</br>Output:</br>".$output);
	}
	fclose($fp);

	print json_encode(array(
			"cmd"  => $cmd,
			"plot" => $plot
		));
	exit;
}
catch (Exception $e) {
	    header($_SERVER['SERVER_PROTOCOL'] . ' 500 Internal Server Error',
	true, 500);
	    print $e->getMessage();
}
?>