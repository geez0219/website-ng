import { Component, OnInit, Inject }  from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

import {MatDialog, MatDialogRef, MAT_DIALOG_DATA} from '@angular/material/dialog';

import { GlobalService } from '../global.service';
import { SearchResultComponent } from '../search-result/search-result.component';

@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements OnInit{
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
              private globalService: GlobalService,
              public dialog: MatDialog) {}
  
  ngOnInit() {}

  search() {
    console.log("searching for " + this.searchText);
    var searchURL = "http://35.165.103.176:3200/search/" + this.searchText;

    this.http.get(searchURL).subscribe(data => {
      console.log(data);
      this.openDialog(data);
    },
    error => {
      console.error(error);
      this.globalService.resetLoading();
      this.router.navigate(['PageNotFound'], {replaceUrl:true})
    });
  }

  openDialog(searchResults): void {
    const dialogRef = this.dialog.open(SearchResultComponent, {
      // height: '400px',
      width: '550px',
      data: searchResults
    });

    dialogRef.afterClosed().subscribe(result => {
      console.log('The dialog was closed');
    });
  }
}
