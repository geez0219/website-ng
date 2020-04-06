import { Component, OnInit } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { Router, ActivatedRoute } from '@angular/router';

import { GlobalService } from '../global.service';

@Component({
  selector: 'app-search-result',
  templateUrl: './search-result.component.html',
  styleUrls: ['./search-result.component.css']
})
export class SearchResultComponent implements OnInit {
  data: any;
  searchText: string;

  structureHeaderDict = {
    'Content-Type': 'application/json',
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }
  structureRequestOptions = {
    headers: new HttpHeaders(this.structureHeaderDict),
  };

  constructor(private http: HttpClient,
              private router: Router,
              private route: ActivatedRoute,
              private globalService: GlobalService) {}

  ngOnInit() {
    this.route.queryParams.subscribe(params => {
        console.log(params.query);
        this.searchText = params.query;

        this.getSearchResults();
    });
  }

  getSearchResults() {
    var searchURL = "https://search.fastestimator.org:3200/search/" + this.searchText;

    this.http.get(searchURL).subscribe(data => {
      console.log(data);
      this.data = data;
    },
    error => {
      console.error(error);
      this.globalService.resetLoading();
      this.router.navigate(['PageNotFound'], {replaceUrl:true})
    });
  }
  onNoClick(): void {

  }

}
